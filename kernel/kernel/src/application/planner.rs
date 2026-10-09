// Copyright (c) 2026 vivo Mobile Communication Co., Ltd.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//       http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! BFS launch planner over the read-only dependency scan.
//!
//! The planner walks the real `DT_NEEDED` graph from the launch root: open
//! each file, scan it with the loader's [`scan_artifact`], classify every
//! dependency request through the namespace rules, and record the images and
//! edges of the whole closure. The output [`NamespaceLoadPlan`] is a transient
//! object describing this launch as observed from the VFS.
//!
//! Its purpose: the launch path sees the complete system closure
//! before linking, so the system permits can be acquired as one atomic batch
//! instead of one-by-one (which would risk an ABBA deadlock between two
//! concurrent sessions).

use alloc::{
    string::{String, ToString},
    vec::Vec,
};

use blueos_loader::{
    scan_artifact, ArtifactIdentity, ArtifactRole, DependencyName, LoadError, LoadErrorKind,
    LoadResult, ScannedArtifact, SessionLimits,
};

use crate::{
    application::{
        adapters::{
            resolver::identity_from_path,
            system_paths::{SystemLibraryEntry, SystemLibraryPaths},
            vfs_reader::VfsElfReader,
        },
        namespace::{resolve_dependency_paths, ApplicationNamespace, DependencyKind, ResolveBase},
    },
    error::code,
    vfs::open_path,
};

/// One planned image: its resolved path, identity and scanned metadata.
pub struct PlannedImage {
    /// The normalized absolute path this image was opened at.
    path: String,
    /// The path-derived graph deduplication key.
    identity: ArtifactIdentity,
    /// Canonical path key when this is a shared system DSO. `None` means the
    /// image belongs only to this application namespace.
    system_key: Option<DependencyName>,
    /// What the scan saw: the optional SONAME and the `DT_NEEDED` set, in
    /// encounter order.
    scanned: ScannedArtifact,
}

impl PlannedImage {
    /// The normalized absolute path this image was planned at.
    pub fn path(&self) -> &str {
        &self.path
    }

    /// The artifact identity used by the linker graph.
    pub fn identity(&self) -> &ArtifactIdentity {
        &self.identity
    }

    /// Whether the planner classified this image as a shared system DSO.
    pub fn system(&self) -> bool {
        self.system_key.is_some()
    }

    /// The canonical registry key for a system image.
    pub fn system_key(&self) -> Option<&DependencyName> {
        self.system_key.as_ref()
    }

    /// The scanned metadata (SONAME + needed, in order).
    pub fn scanned(&self) -> &ScannedArtifact {
        &self.scanned
    }
}

/// One dependency edge: requester → provider.
pub struct PlannedEdge {
    /// The index of the requesting image in the plan.
    requester: usize,
    /// The raw `DT_NEEDED` string as scanned (before classification).
    request: DependencyName,
    /// The index of the image the request resolved to.
    provider: usize,
}

impl PlannedEdge {
    /// The requesting image's index.
    pub fn requester(&self) -> usize {
        self.requester
    }

    /// The raw request string.
    pub fn request(&self) -> &DependencyName {
        &self.request
    }

    /// The provider image's index.
    pub fn provider(&self) -> usize {
        self.provider
    }
}

/// The transient launch plan: every image in the closure, every edge, and the
/// deduplicated system catalog keys the launch will batch-acquire.
pub struct NamespaceLoadPlan {
    images: Vec<PlannedImage>,
    edges: Vec<PlannedEdge>,
    /// The sorted, deduplicated system catalog paths the closure touches.
    system_keys: Vec<DependencyName>,
}

impl NamespaceLoadPlan {
    /// Every planned image, in BFS discovery order.
    pub fn images(&self) -> &[PlannedImage] {
        &self.images
    }

    /// Every dependency edge, in BFS order.
    pub fn edges(&self) -> &[PlannedEdge] {
        &self.edges
    }

    /// The system catalog keys (sorted, deduplicated) this launch needs:
    /// the batch-acquire input.
    pub fn system_keys(&self) -> &[DependencyName] {
        &self.system_keys
    }
}

/// The BFS planner: scan the real dependency closure of one launch.
pub struct NamespaceLoadPlanner<'a> {
    namespace: &'a ApplicationNamespace,
    system_catalog: &'static SystemLibraryPaths,
    limits: SessionLimits,
}

impl<'a> NamespaceLoadPlanner<'a> {
    /// Plan over `namespace`'s frozen launch state, the system catalog and
    /// the session limits.
    pub fn new(
        namespace: &'a ApplicationNamespace,
        system_catalog: &'static SystemLibraryPaths,
        limits: SessionLimits,
    ) -> Self {
        Self {
            namespace,
            system_catalog,
            limits,
        }
    }

    /// Walk the closure from the namespace's root path.
    ///
    /// Each discovered image is scanned read-only and assigned an identity
    /// derived from its normalized path.
    pub fn plan(&self) -> LoadResult<NamespaceLoadPlan> {
        self.plan_from(self.namespace.root_path(), ArtifactRole::ExecutableRoot)
    }

    /// Scan a runtime DSO without changing the application's search directory.
    pub fn plan_shared(&self, path: &str) -> LoadResult<NamespaceLoadPlan> {
        self.plan_from(path, ArtifactRole::SharedObject)
    }

    fn plan_from(&self, root_path: &str, role: ArtifactRole) -> LoadResult<NamespaceLoadPlan> {
        let mut images: Vec<PlannedImage> = Vec::new();
        let mut edges: Vec<PlannedEdge> = Vec::new();
        let mut system_keys: Vec<DependencyName> = Vec::new();

        // Seed the root: an exact open, no fallback. The BFS queue
        // pairs each image with its depth so the session's
        // dependency-depth limit bounds the longest chain, not the image
        // count.
        let mut queue: Vec<(usize, u16)> = Vec::new();
        let mut queue_head = 0;
        let root = self.open_and_scan(root_path, role)?;
        if let Some(key) = root.system_key() {
            system_keys.push(key.clone());
        }
        images.push(root);
        queue.push((0, 1));

        while queue_head < queue.len() {
            let (requester_index, depth) = queue[queue_head];
            queue_head += 1;
            // The requester's own directory: relative DT_NEEDED resolve
            // against the requesting ELF's directory.
            let requester_dir = parent_dir(&images[requester_index].path);
            let requester_is_system = images[requester_index].system();
            let needed: Vec<DependencyName> = images[requester_index].scanned().needed.to_vec();
            for request in needed {
                self.limits
                    .check_dependency_edge_count((edges.len() + 1) as u32)?;
                let (provider_index, is_new) = self.resolve_request(
                    &mut images,
                    &mut system_keys,
                    &requester_dir,
                    requester_is_system,
                    &request,
                )?;
                edges.push(PlannedEdge {
                    requester: requester_index,
                    request,
                    provider: provider_index,
                });
                if is_new {
                    let next_depth = depth + 1;
                    self.limits.check_dependency_depth(next_depth)?;
                    queue.push((provider_index, next_depth));
                }
            }
        }

        system_keys.sort();
        system_keys.dedup();
        self.limits.check_image_count(images.len() as u32)?;
        Ok(NamespaceLoadPlan {
            images,
            edges,
            system_keys,
        })
    }

    /// Resolve one dependency request to a planned image index, opening and
    /// scanning the file on first sight.
    ///
    /// Returns the provider's index in `images` and whether the image was
    /// discovered by this request (`true`) or already planned (`false`); on
    /// success a newly discovered image has been appended to `images` and,
    /// when it is a system DSO, its catalog path added to `system_keys`.
    fn resolve_request(
        &self,
        images: &mut Vec<PlannedImage>,
        system_keys: &mut Vec<DependencyName>,
        requester_dir: &str,
        requester_is_system: bool,
        request: &DependencyName,
    ) -> LoadResult<(usize, bool)> {
        let request_str = core::str::from_utf8(request.as_bytes()).map_err(|_| {
            LoadError::new(
                LoadErrorKind::BadElf,
                blueos_loader::ErrorContext::Dependency {
                    requester: 0,
                    needed: request.as_bytes().into(),
                },
            )
        })?;
        let kind = DependencyKind::classify(request_str);

        // A system DSO requester never resolves against the application
        // namespace: its relocations must be identical no matter
        // which application first triggered its load.
        let candidates: Vec<String> = if requester_is_system {
            match kind {
                DependencyKind::Absolute | DependencyKind::Relative => {
                    // Path requests from a system DSO must hit the catalog.
                    let mut resolved = resolve_dependency_paths(
                        self.namespace,
                        ResolveBase::RequesterDirectory(requester_dir),
                        request_str,
                    );
                    resolved.retain(|path| self.system_catalog.resolve_path(path).is_some());
                    resolved
                }
                DependencyKind::PlainName => {
                    match self.system_catalog.resolve_name(request.as_bytes()) {
                        Some(entry) => Vec::from([entry.path.to_string()]),
                        None => Vec::new(),
                    }
                }
            }
        } else {
            match kind {
                DependencyKind::Absolute | DependencyKind::Relative => resolve_dependency_paths(
                    self.namespace,
                    ResolveBase::RequesterDirectory(requester_dir),
                    request_str,
                ),
                DependencyKind::PlainName => {
                    // Private lib first; fall through to the catalog on
                    // ENOENT only.
                    let private = resolve_dependency_paths(
                        self.namespace,
                        ResolveBase::RequesterDirectory(requester_dir),
                        request_str,
                    );
                    if !self.first_is_missing(&private) {
                        private
                    } else {
                        match self.system_catalog.resolve_name(request.as_bytes()) {
                            Some(entry) => Vec::from([entry.path.to_string()]),
                            // Keep the private candidate so the open reports
                            // the actual ENOENT path.
                            None => private,
                        }
                    }
                }
            }
        };

        if candidates.is_empty() {
            return Err(unresolved(request));
        }

        // Try candidates in order: a missing file advances to the next; a
        // present-but-unloadable file fails without fallback.
        let mut last_error = unresolved(request);
        for candidate in candidates {
            // Path dedup: the same resolved path is the same image.
            if let Some(index) = images.iter().position(|image| image.path == candidate) {
                return Ok((index, false));
            }
            match self.open_and_scan(&candidate, ArtifactRole::SharedObject) {
                Ok(image) => {
                    if let Some(index) = images.iter().position(|existing| {
                        existing.identity == image.identity && existing.system() == image.system()
                    }) {
                        return Ok((index, false));
                    }
                    let system_key = image.system_key.clone();
                    images.push(image);
                    if let Some(key) = system_key {
                        system_keys.push(key);
                    }
                    return Ok((images.len() - 1, true));
                }
                Err(error) => {
                    if self.is_enoent(&candidate) {
                        last_error = error;
                        continue;
                    }
                    return Err(error);
                }
            }
        }
        Err(last_error)
    }

    /// Open `path` and scan it read-only. `role` is `ExecutableRoot` for the launch root and
    /// `SharedObject` for every dependency.
    fn open_and_scan(&self, path: &str, role: ArtifactRole) -> LoadResult<PlannedImage> {
        let file = open_path(path, libc::O_RDONLY, 0).map_err(|_| backend_error())?;
        let reader = VfsElfReader::new(file);
        let scanned = scan_artifact(
            &reader,
            self.namespace.profile(),
            role,
            *self.limits.per_image(),
        )?;
        // A path that equals a catalog entry's path is a shared system DSO
        // whatever DT_SONAME it carries.
        let entry = if role == ArtifactRole::SharedObject {
            self.system_catalog.resolve_path(path)
        } else {
            None
        };
        let identity = identity_from_path(path);
        Ok(PlannedImage {
            path: path.to_string(),
            identity,
            system_key: entry.map(SystemLibraryEntry::key).transpose()?,
            scanned,
        })
    }

    /// Whether the first private candidate is absent. Only `ENOENT` and
    /// `ENOTDIR` permit a system fallback; every other open error keeps the
    /// private candidate selected so the launch fails on that exact path.
    fn first_is_missing(&self, paths: &[String]) -> bool {
        match paths.first() {
            Some(path) => matches!(
                open_path(path, libc::O_RDONLY, 0),
                Err(error) if error == code::ENOENT || error == code::ENOTDIR
            ),
            None => true,
        }
    }

    /// Whether an open of `path` fails with the "not present" errno family
    /// (: only `ENOENT`/`ENOTDIR` advance the search).
    fn is_enoent(&self, path: &str) -> bool {
        matches!(
            open_path(path, libc::O_RDONLY, 0),
            Err(error) if error == code::ENOENT || error == code::ENOTDIR
        )
    }
}

fn unresolved(request: &DependencyName) -> LoadError {
    LoadError::new(
        LoadErrorKind::Backend,
        blueos_loader::ErrorContext::Dependency {
            requester: 0,
            needed: request.as_bytes().into(),
        },
    )
}

fn backend_error() -> LoadError {
    LoadError::new(LoadErrorKind::Backend, blueos_loader::ErrorContext::None)
}

/// The parent directory of an absolute normalized path: everything before
/// the final `/` component, or `/` for top-level entries.
fn parent_dir(path: &str) -> String {
    let trimmed = path.trim_end_matches('/');
    match trimmed.rfind('/') {
        Some(0) => "/".to_string(),
        Some(at) => trimmed[..at].to_string(),
        None => "/".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use alloc::{vec, vec::Vec};
    use blueos_loader::SessionLimits;
    use blueos_test_macro::test;
    #[cfg(use_defmt)]
    use defmt::println;
    #[cfg(not(use_defmt))]
    use semihosting::println;

    use super::*;
    use crate::application::{
        adapters::system_paths::SystemLibraryEntry, namespace::ApplicationNamespace,
    };

    /// The test catalog mirrors the boot runtime's: two system DSOs
    /// keyed by name and path.
    static TEST_CATALOG: SystemLibraryPaths = SystemLibraryPaths::new(&[
        SystemLibraryEntry {
            lookup_name: b"libc.so.1",
            path: "/system/lib/libc.so.1",
            keep_cached: true,
        },
        SystemLibraryEntry {
            lookup_name: b"libsys_extra.so.1",
            path: "/system/lib/libsys_extra.so.1",
            keep_cached: false,
        },
    ]);

    /// Build a minimal DSO matching the board's ELF class and machine.
    /// One PT_LOAD covers the headers, PT_DYNAMIC and its string table.
    fn dso_elf(needed: &[&str], soname: Option<&str>) -> Vec<u8> {
        let profile = crate::application::board_dynamic_profile();
        let elf64 = profile.class() == blueos_loader::ElfClass::Elf64;
        let word = if elf64 { 8 } else { 4 };
        let ehdr = if elf64 { 64 } else { 52 };
        let phdr = if elf64 { 56 } else { 32 };
        let dyn_off = ehdr + 2 * phdr;
        let dyn_len = (3 + needed.len() + usize::from(soname.is_some())) * 2 * word;
        let str_off = dyn_off + dyn_len;
        let mut dynstr = vec![0];
        let mut entries = vec![(DT_STRTAB, (0x1000 + str_off) as u64), (DT_STRSZ, 0)];
        for (index, name) in needed.iter().chain(soname.iter()).enumerate() {
            let tag = if index < needed.len() {
                DT_NEEDED
            } else {
                DT_SONAME
            };
            entries.push((tag, dynstr.len() as u64));
            dynstr.extend_from_slice(name.as_bytes());
            dynstr.push(0);
        }
        entries[1].1 = dynstr.len() as u64;
        let file_len = str_off + dynstr.len();
        let mut bytes = vec![0u8; file_len];
        bytes[..4].copy_from_slice(b"\x7fELF");
        bytes[4] = if elf64 { 2 } else { 1 };
        bytes[5] = 1;
        bytes[6] = 1;
        let mut put = |at: usize, value: u64, width: usize| {
            bytes[at..at + width].copy_from_slice(&value.to_le_bytes()[..width]);
        };
        put(16, 3, 2); // ET_DYN
        put(18, u64::from(profile.machine()), 2);
        put(20, 1, 4);
        put(if elf64 { 32 } else { 28 }, ehdr as u64, word);
        let sizes = if elf64 { 52 } else { 40 };
        put(sizes, ehdr as u64, 2);
        put(sizes + 2, phdr as u64, 2);
        put(sizes + 4, 2, 2);
        for (index, (kind, offset, len, flags)) in [(1, 0, file_len, 5), (2, dyn_off, dyn_len, 4)]
            .into_iter()
            .enumerate()
        {
            let at = ehdr + index * phdr;
            put(at, kind, 4);
            if elf64 {
                put(at + 4, flags, 4);
            } else {
                put(at + 24, flags, 4);
            }
            let fields = at + if elf64 { 8 } else { 4 };
            for (index, value) in [offset, 0x1000 + offset, 0x1000 + offset, len, len]
                .into_iter()
                .enumerate()
            {
                put(fields + index * word, value as u64, word);
            }
            put(at + if elf64 { 48 } else { 28 }, word as u64, word);
        }
        for (index, (tag, value)) in entries.into_iter().enumerate() {
            let at = dyn_off + index * 2 * word;
            put(at, u64::from(tag), word);
            put(at + word, value, word);
        }
        bytes[str_off..].copy_from_slice(&dynstr);
        bytes
    }

    const DT_STRTAB: u32 = 5;
    const DT_STRSZ: u32 = 10;
    const DT_NEEDED: u32 = 1;
    const DT_SONAME: u32 = 14;

    /// Copy `path` into a NUL-terminated stack buffer: the VFS syscall
    /// helpers take raw C paths, and a Rust `&str` is not NUL-terminated.
    fn c_path<const N: usize>(path: &str) -> [core::ffi::c_char; N] {
        assert!(path.len() < N, "test path too long: {path}");
        let mut buffer = [0 as core::ffi::c_char; N];
        for (byte, slot) in path.bytes().zip(buffer.iter_mut()) {
            *slot = byte as core::ffi::c_char;
        }
        buffer
    }

    fn write_file(path: &str, bytes: &[u8]) {
        let c_path = c_path::<64>(path);
        let fd = crate::vfs::syscalls::open(
            c_path.as_ptr(),
            libc::O_CREAT | libc::O_WRONLY | libc::O_TRUNC,
            0o644,
        );
        assert!(fd > 0, "open {path} for write: {fd}");
        let wrote = crate::vfs::syscalls::write(fd, bytes.as_ptr(), bytes.len());
        assert_eq!(wrote as usize, bytes.len(), "write {path}");
        crate::vfs::syscalls::close(fd);
    }

    fn make_dir(path: &str) {
        let c_path = c_path::<64>(path);
        let rc = crate::vfs::syscalls::mkdir(c_path.as_ptr(), 0o755);
        // Tests share the root tmpfs; a directory a previous test seeded is
        // fine to reuse, exactly as the boot seeder tolerates EEXIST.
        assert!(rc == 0 || rc == -libc::EEXIST, "mkdir {path}: {rc}");
    }

    /// Remove a file, tolerating ENOENT (a fresh tmpfs never had it).
    fn remove_file(path: &str) {
        let c_path = c_path::<64>(path);
        let rc = crate::vfs::syscalls::unlink(c_path.as_ptr());
        assert!(rc == 0 || rc == -libc::ENOENT, "unlink {path}: {rc}");
    }

    fn namespace_at(pwd: &str, launch: &str) -> ApplicationNamespace {
        ApplicationNamespace::from_launch_path(
            launch,
            pwd,
            crate::application::board_dynamic_profile(),
        )
        .expect("namespace")
    }

    fn seed_layout() {
        make_dir("/apps");
        make_dir("/apps/hello");
        make_dir("/apps/hello/lib");
        make_dir("/system");
        make_dir("/system/lib");
        // Tests share the tmpfs: clear any leftovers earlier tests planted in
        // the private lib dir, so plain-name resolution starts from a clean
        // slate.
        remove_file("/apps/hello/lib/libsys_extra.so.1");
        remove_file("/apps/hello/lib/libmissing.so.1");
        // The root: needs a private DSO by plain name and the system libc.
        write_file(
            "/apps/hello/app.elf",
            &dso_elf(&["libfoo.so.1", "libc.so.1"], Some("app.elf")),
        );
        // The private DSO: no SONAME, plain-name request only.
        write_file("/apps/hello/lib/libfoo.so.1", &dso_elf(&[], None));
        // The system DSOs.
        write_file("/system/lib/libc.so.1", &dso_elf(&[], Some("libc.so.1")));
        write_file(
            "/system/lib/libsys_extra.so.1",
            &dso_elf(&[], Some("libsys_extra.so.1")),
        );
    }

    /// Plan the hello layout and assert the BFS closure, edges and system
    /// keys (paths matrix).
    #[test]
    fn plan_walks_private_then_system_closure() {
        seed_layout();
        let namespace = namespace_at("/apps/hello", "app.elf");
        let planner = NamespaceLoadPlanner::new(&namespace, &TEST_CATALOG, SessionLimits::DEFAULT);
        let plan = planner.plan().expect("plan");

        // root, private DSO, system DSO
        assert_eq!(plan.images().len(), 3);
        assert_eq!(plan.images()[0].path(), "/apps/hello/app.elf");
        assert!(!plan.images()[0].system());
        assert_eq!(plan.images()[1].path(), "/apps/hello/lib/libfoo.so.1");
        assert!(!plan.images()[1].system());
        assert_eq!(plan.images()[2].path(), "/system/lib/libc.so.1");
        assert!(plan.images()[2].system());

        // Edges: root→libfoo (plain name, private hit), root→libc (plain
        // name, catalog fallback).
        assert_eq!(plan.edges().len(), 2);
        assert_eq!(plan.edges()[0].requester(), 0);
        assert_eq!(plan.edges()[0].provider(), 1);
        assert_eq!(plan.edges()[1].requester(), 0);
        assert_eq!(plan.edges()[1].provider(), 2);

        // System keys: sorted, deduplicated, just libc.
        assert_eq!(plan.system_keys().len(), 1);
        assert_eq!(plan.system_keys()[0].as_bytes(), b"/system/lib/libc.so.1");

        // The scanned SONAME came through.
        assert_eq!(
            plan.images()[0]
                .scanned()
                .declared_soname
                .as_ref()
                .unwrap()
                .as_bytes(),
            b"app.elf"
        );
        assert!(plan.images()[1].scanned().declared_soname.is_none());
    }

    /// A relative DT_NEEDED from the root resolves against the requester
    /// ELF's directory.
    #[test]
    fn plan_resolves_relative_needed_against_requester_dir() {
        seed_layout();
        // root needs "./lib/libfoo.so.1" by relative path.
        write_file(
            "/apps/hello/app.elf",
            &dso_elf(&["./lib/libfoo.so.1"], Some("app.elf")),
        );
        let namespace = namespace_at("/apps/hello", "app.elf");
        let planner = NamespaceLoadPlanner::new(&namespace, &TEST_CATALOG, SessionLimits::DEFAULT);
        let plan = planner.plan().expect("plan");
        assert_eq!(plan.images().len(), 2);
        assert_eq!(plan.images()[1].path(), "/apps/hello/lib/libfoo.so.1");
        assert!(plan.system_keys().is_empty());
    }

    /// An absolute `DT_NEEDED` that equals a catalog path is a system DSO.
    #[test]
    fn plan_treats_catalog_path_needed_as_system() {
        seed_layout();
        write_file(
            "/apps/hello/app.elf",
            &dso_elf(&["/system/lib/libc.so.1"], Some("app.elf")),
        );
        let namespace = namespace_at("/apps/hello", "app.elf");
        let planner = NamespaceLoadPlanner::new(&namespace, &TEST_CATALOG, SessionLimits::DEFAULT);
        let plan = planner.plan().expect("plan");
        assert_eq!(plan.images().len(), 2);
        assert!(plan.images()[1].system());
        assert_eq!(plan.system_keys()[0].as_bytes(), b"/system/lib/libc.so.1");
    }

    /// A plain-name request whose private file is missing falls back to the
    /// catalog.
    #[test]
    fn plan_falls_back_to_catalog_when_private_missing() {
        seed_layout();
        write_file(
            "/apps/hello/app.elf",
            &dso_elf(&["libsys_extra.so.1"], Some("app.elf")),
        );
        // Ensure no private copy shadows the name: a leftover garbage file
        // from an earlier test would make the plan fail on it instead.
        remove_file("/apps/hello/lib/libsys_extra.so.1");
        let namespace = namespace_at("/apps/hello", "app.elf");
        let planner = NamespaceLoadPlanner::new(&namespace, &TEST_CATALOG, SessionLimits::DEFAULT);
        let plan = planner.plan().expect("plan");
        assert_eq!(plan.images().len(), 2);
        assert_eq!(plan.images()[1].path(), "/system/lib/libsys_extra.so.1");
        assert!(plan.images()[1].system());
        assert_eq!(
            plan.system_keys()[0].as_bytes(),
            b"/system/lib/libsys_extra.so.1"
        );
    }

    /// A plain-name request whose private file is present but broken fails
    /// without falling back.
    #[test]
    fn plan_broken_private_file_does_not_fall_back() {
        seed_layout();
        write_file(
            "/apps/hello/app.elf",
            &dso_elf(&["libsys_extra.so.1"], Some("app.elf")),
        );
        // A garbage private file shadows the catalog entry by name.
        write_file("/apps/hello/lib/libsys_extra.so.1", &[0u8; 16]);
        let namespace = namespace_at("/apps/hello", "app.elf");
        let planner = NamespaceLoadPlanner::new(&namespace, &TEST_CATALOG, SessionLimits::DEFAULT);
        let result = planner.plan();
        assert!(
            result.is_err(),
            "a present-but-broken private DSO must fail the plan without catalog fallback"
        );
    }

    /// A plain-name request resolved nowhere is unresolved.
    #[test]
    fn plan_unresolved_plain_name_fails() {
        seed_layout();
        write_file(
            "/apps/hello/app.elf",
            &dso_elf(&["libmissing.so.1"], Some("app.elf")),
        );
        let namespace = namespace_at("/apps/hello", "app.elf");
        let planner = NamespaceLoadPlanner::new(&namespace, &TEST_CATALOG, SessionLimits::DEFAULT);
        let result = planner.plan();
        assert!(
            result.is_err(),
            "an unresolvable DT_NEEDED must fail the plan"
        );
    }

    /// The same DSO requested twice is planned only once.
    #[test]
    fn plan_dedups_repeated_requests() {
        seed_layout();
        write_file(
            "/apps/hello/app.elf",
            &dso_elf(&["libc.so.1", "libc.so.1"], Some("app.elf")),
        );
        let namespace = namespace_at("/apps/hello", "app.elf");
        let planner = NamespaceLoadPlanner::new(&namespace, &TEST_CATALOG, SessionLimits::DEFAULT);
        let plan = planner.plan().expect("plan");
        assert_eq!(plan.images().len(), 2);
        assert_eq!(plan.edges().len(), 2);
        // Both edges point at the same provider.
        assert_eq!(plan.edges()[0].provider(), plan.edges()[1].provider());
        assert_eq!(plan.system_keys().len(), 1);
    }

    /// A plain name and an explicit relative path that normalize to the same
    /// VFS path produce two graph edges but only one mapped image.
    #[test]
    fn plan_dedups_lookup_aliases_by_path() {
        seed_layout();
        write_file(
            "/apps/hello/app.elf",
            &dso_elf(&["libfoo.so.1", "./lib/libfoo.so.1"], Some("app.elf")),
        );
        let namespace = namespace_at("/apps/hello", "app.elf");
        let planner = NamespaceLoadPlanner::new(&namespace, &TEST_CATALOG, SessionLimits::DEFAULT);
        let plan = planner.plan().expect("plan");
        assert_eq!(plan.images().len(), 2);
        assert_eq!(plan.edges().len(), 2);
        assert_eq!(plan.edges()[0].provider(), plan.edges()[1].provider());
    }
}
