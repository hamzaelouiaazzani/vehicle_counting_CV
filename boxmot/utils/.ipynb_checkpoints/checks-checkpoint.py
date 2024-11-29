# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

import subprocess

import pkg_resources

from boxmot.utils import REQUIREMENTS, logger


class TestRequirements():

    def check_requirements(self):
        requirements = pkg_resources.parse_requirements(REQUIREMENTS.open())
        self.check_packages(requirements)

    # def check_packages(self, requirements, cmds=''):
    #     """Test that each required package is available."""
    #     # Ref: https://stackoverflow.com/a/45474387/

    #     s = ''  # missing packages
    #     for r in requirements:
    #         r = str(r)
    #         try:
    #             pkg_resources.require(r)
    #         except Exception as e:
    #             logger.error(f'{e}')
    #             s += f'"{r}" '
    #     if s:
    #         logger.warning(f'\nMissing packages: {s}\nAtempting installation...')
    #         try:
    #             subprocess.check_output(f'pip install --no-cache {s} {cmds}', shell=True, stderr=subprocess.STDOUT)
    #         except Exception as e:
    #             logger.error(e)
    #             exit()
    #         logger.success('All the missing packages were installed successfully')


    def check_packages(self, requirements, cmds=''):
        """Test that each required package is available."""
        missing_packages = []
        
        print(f"requirements are: {requirements}")
        
        for r in requirements:
            r = str(r)
            if os.path.isdir(r):  # Check if the requirement is a local directory
                try:
                    subprocess.check_output(f'pip install {r}', shell=True, stderr=subprocess.STDOUT)
                except Exception as e:
                    logger.error(f'Failed to install package from {r}: {e}')
                    exit()
            else:
                try:
                    pkg_resources.require(r)
                except Exception as e:
                    logger.error(f'{e}')
                    missing_packages.append(r)
        if missing_packages:
            logger.warning(f'\nMissing packages: {" ".join(missing_packages)}\nAttempting installation...')
            try:
                subprocess.check_output(f'pip install --no-cache {" ".join(missing_packages)} {cmds}', shell=True, stderr=subprocess.STDOUT)
            except Exception as e:
                logger.error(e)
                exit()
            logger.success('All the missing packages were installed successfully')

