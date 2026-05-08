import logging
from pathlib import Path
from typing import List

from .utils import UploadTracker

logger = logging.getLogger(__name__)

class ExoFOPPackager:
    """Handles the collection and packaging of ExoFOP data products."""

    def __init__(self, targets: List[str] | None = None):
        self.base_dir = Path.cwd()
        self.data_dir = self.base_dir / "organized_data"
        self.logs_dir = self.base_dir / "logs"
        self.tmp_dir = self.base_dir / "tmp" / "exofop_uploads"
        
        self.targets = [t.lower() for t in targets] if targets else []
        self.tracker = UploadTracker(self.logs_dir)
        
    def run_prepare(self) -> None:
        """Scan directories and prepare uploads for pending target/date combinations."""
        if not self.data_dir.exists():
            logger.error(f"Data directory {self.data_dir} does not exist.")
            return

        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        found_any = False

        for obj_folder in sorted(self.data_dir.iterdir()):
            if not obj_folder.is_dir():
                continue
                
            target_name = obj_folder.name
            if self.targets and target_name.lower() not in self.targets:
                continue

            exofop_dir = obj_folder / "exofop"
            if not exofop_dir.exists():
                logger.info(f"No exofop data found for {target_name}. Skipping.")
                continue

            for date_folder in sorted(exofop_dir.iterdir()):
                if not date_folder.is_dir():
                    continue

                local_date = date_folder.name
                
                # Check status
                status = self.tracker.get_status(target_name, local_date)
                if status in ("prepared", "uploaded"):
                    logger.info(f"Target {target_name} on {local_date} is already {status}. Skipping.")
                    continue
                    
                found_any = True
                logger.info(f"Scanning target {target_name} on {local_date}...")
                self._process_date(target_name, local_date, date_folder)

        if not found_any:
            logger.info("No new data found to prepare.")

    def _process_date(self, target_name: str, local_date: str, date_folder: Path) -> None:
        """Process a specific date folder for a target."""
        logger.info(f"Found pending upload for {target_name} on {local_date} at {date_folder}")
        
        print(f"\n[{target_name} | {local_date}]")
        data_tag = input(f"Ingresa el Data Tag para {target_name} de la fecha {local_date}: ").strip()
        
        if not data_tag:
            logger.warning(f"No Data Tag provided. Skipping {target_name} on {local_date}.")
            return

        collected_files = []
        for file_path in date_folder.rglob("*"):
            if not file_path.is_file():
                continue
                
            # Ignorar completamente la carpeta AIJ
            if "AIJ" in file_path.parts:
                continue
                
            name = file_path.name
            if name.startswith("."):
                continue
            if name.endswith(".plotcfg") or name.endswith(".apertures"):
                continue
            if name.endswith("exofop_metadata.txt"):
                continue
            if "fitpanel" in name.lower():
                continue
                
            collected_files.append(file_path)
            
        if not collected_files:
            logger.warning(f"No files found to upload for {target_name} on {local_date}.")
            return
            
        logger.info(f"Collected {len(collected_files)} files for {target_name}.")
        self._package_files(target_name, local_date, data_tag, collected_files)

    def _package_files(self, target_name: str, local_date: str, data_tag: str, files: List[Path]) -> None:
        date_compact = local_date.replace("-", "")
        
        # Find next available sequence number for the prefix spYYYYMMDD-nnn
        prefix = "sp"
        seq = 1
        for i in range(1, 1000):
            if not (self.tmp_dir / f"{prefix}{date_compact}-{i:03d}.tar").exists():
                seq = i
                break
                
        base_name = f"{prefix}{date_compact}-{seq:03d}"
        descriptor_name = f"{base_name}.txt"
        descriptor_path = self.tmp_dir / descriptor_name
        
        lines = []
        for f in files:
            desc = self._get_description(f.name)
            # Format: FileName|DataTag|Group|ProprietaryPeriod|Description
            line = f"{f.name}|{data_tag}|tfopwg|12|{desc}"
            lines.append(line)
            
        try:
            with open(descriptor_path, "w") as out:
                out.write("\n".join(lines) + "\n")
            logger.info(f"Generated descriptor file {descriptor_name}")
        except Exception as e:
            logger.error(f"Failed to create descriptor file {descriptor_path}: {e}")
            return
            
        self._create_tarball(target_name, local_date, descriptor_path, files)

    def _get_description(self, file_name: str) -> str:
        """Map filenames to ExoFOP SG1 descriptions."""
        lower_name = file_name.lower()
        if "notes.txt" in lower_name:
            return "Observing Notes"
        elif "field" in lower_name and lower_name.endswith((".png", ".pdf")):
            return "Field of View"
        elif "lightcurve" in lower_name and lower_name.endswith((".png", ".pdf")):
            return "Light Curve Plot"
        elif "profile" in lower_name and lower_name.endswith((".png", ".pdf")):
            return "Seeing Profile"
        elif "comparison" in lower_name and lower_name.endswith((".png", ".pdf")):
            return "Comparison Star Plot"
        elif lower_name.endswith(".fits"):
            return "WCS FITS Image"
        elif lower_name.endswith((".csv", ".tbl")):
            return "Measurement Table"
        return "Data Product"

    def _create_tarball(self, target_name: str, local_date: str, descriptor: Path, files: List[Path]) -> None:
        import tarfile
        tar_name = descriptor.with_suffix(".tar")
        
        try:
            with tarfile.open(tar_name, "w") as tar:
                # ExoFOP requires a single flat directory with no subdirectories
                tar.add(descriptor, arcname=descriptor.name)
                for f in files:
                    tar.add(f, arcname=f.name)
                    
            logger.info(f"Successfully packaged {len(files) + 1} files into {tar_name.name}")
            
            # Update the tracking JSON
            self.tracker.update_status(target_name, local_date, "prepared", tar_file=tar_name.name)
            
        except Exception as e:
            logger.error(f"Failed to create tarball {tar_name}: {e}")
