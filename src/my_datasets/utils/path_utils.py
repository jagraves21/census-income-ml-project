import dataclasses
import pathlib
from typing import Optional


def print_tree(startpath, show_hidden=False):
	startpath = pathlib.Path(startpath).resolve()
	print(startpath.name or str(startpath))

	def _walk(current_path, prefix=""):
		try:
			entries = list(current_path.iterdir())
		except PermissionError:
			print(prefix + "└── [permission denied]")
			return

		if not show_hidden:
			entries = [
				entry
				for entry in entries
				if not entry.name.startswith(".")
			]

		# get metadata once
		entries_meta = []
		for entry in entries:
			try:
				is_dir = entry.is_dir()
				is_symlink = entry.is_symlink()
			except OSError:
				entries_meta.append((entry, None, None))
				continue

			entries_meta.append((entry, is_dir, is_symlink))

		# sort: files first, then directories, then name
		entries_meta.sort(
			key=lambda x: (x[1] is not True, x[0].name.lower())
		)

		for i, (path, is_dir, is_symlink) in enumerate(entries_meta):
			is_last = i == len(entries_meta) - 1
			branch = "└── " if is_last else "├── "

			if is_dir is None:
				print(prefix + branch + "[error reading entry]")
				continue

			if is_symlink and not path.exists():
				print(prefix + branch + path.name + " -> [broken link]")
				continue

			label = path.name + ("/" if is_dir and not is_symlink else "")
			print(prefix + branch + label)

			if is_dir and not is_symlink:
				next_prefix = prefix + ("    " if is_last else "│   ")
				_walk(path, next_prefix)

	_walk(startpath)


@dataclasses.dataclass
class ProjectPaths:
	project_root: pathlib.Path | None = None
	data_folder_name: str = "data"
	markers: str = "src"

	project_name: str = dataclasses.field(init=False)
	data_folder: pathlib.Path = dataclasses.field(init=False)

	def __post_init__(self):
		start_path = self._get_start_path()

		root = self._resolve_root(start_path)

		self.project_root = root
		self.project_name = root.name
		self.data_folder = root / self.data_folder_name

	def _get_start_path(self):
		try:
			return pathlib.Path(__file__).resolve()
		except NameError:
			return pathlib.Path.cwd().resolve()

	def _resolve_root(self, start_path):
		if self.project_root is not None:
			return pathlib.Path(self.project_root).expanduser().resolve()

		for strategy in (self._find_by_markers,):
			result = strategy(start_path)
			if result is not None:
				return result

		raise FileNotFoundError("Could not determine project root.")

	def _find_by_markers(self, start_path):
		marker = self._normalize_markers()
		for path in (start_path, *start_path.parents):
			if (path / marker).exists():
				return path
		return None

	def _normalize_markers(self):
		if self.markers is None:
			return "src"

		if isinstance(self.markers, str):
			if self.markers.strip() == "":
				raise ValueError("markers cannot be an empty string")
			return self.markers

		raise TypeError("markers must be a non-empty string (e.g. 'src')")

	def get_data_dir(self):
		 data_dir = self.data_folder
		 data_dir.mkdir(parents=True, exist_ok=True)
		 return data_dir

