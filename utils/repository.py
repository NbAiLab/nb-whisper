import subprocess

from huggingface_hub import Repository as HFRepository
from huggingface_hub.repository import _lfs_log_progress, logger
from huggingface_hub.utils import run_subprocess


class Repository(HFRepository):

    def lfs_prune(self, recent=True):
        """
        git lfs prune

        Args:
            recent (`bool`, *optional*, defaults to `True`):
                Whether to prune files even if they were referenced by recent
                commits. See the following
                [link](https://github.com/git-lfs/git-lfs/blob/f3d43f0428a84fc4f1e5405b76b5a73ec2437e65/docs/man/git-lfs-prune.1.ronn#recent-files)
                for more information.
        """
        try:
            with _lfs_log_progress():
                result = run_subprocess(f"git lfs prune {'--recent' if recent else ''}", self.local_dir)
                logger.info(result.stdout)
        except subprocess.CalledProcessError as exc:
            raise EnvironmentError(exc.stderr)