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
            if "transit_times.dat" in name.lower():
                continue
                
            collected_files.append(file_path)
            
        if not collected_files:
            logger.warning(f"No files found to upload for {target_name} on {local_date}.")
            return
            
        logger.info(f"Collected {len(collected_files)} files for {target_name}.")
        self._package_files(target_name, local_date, data_tag, collected_files)

    def _package_files(self, target_name: str, local_date: str, data_tag: str, files: List[Path]) -> None:
        date_compact = local_date.replace("-", "")
        
        # Find next available sequence number
        prefix = "pa"
        seq = 1
        for i in range(1, 1000):
            if not (self.tmp_dir / f"{prefix}{date_compact}-{i:03d}.tar").exists():
                seq = i
                break
                
        tar_name = f"{prefix}{date_compact}-{seq:03d}.tar"
        tar_path = self.tmp_dir / tar_name
        
        # Prepare metadata JSON for the API uploader
        import json
        metadata = {
            "target_name": target_name,
            "data_tag": data_tag,
            "local_date": local_date
        }
        meta_path = self.tmp_dir / "upload_metadata.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f)
        
        import tarfile
        try:
            with tarfile.open(tar_path, "w") as tar:
                # Add metadata file
                tar.add(meta_path, arcname="upload_metadata.json")
                
                # Add all scientific files with their ORIGINAL names
                for file_path in files:
                    tar.add(file_path, arcname=file_path.name)
                    
            logger.info(f"Packaged {len(files)} files into {tar_path}")
            # Mark as prepared
            self.tracker.update_status(target_name, local_date, "prepared", tar_file=tar_name)
        except Exception as e:
            logger.error(f"Failed to create tarball {tar_path}: {e}")
        finally:
            if meta_path.exists():
                meta_path.unlink()
