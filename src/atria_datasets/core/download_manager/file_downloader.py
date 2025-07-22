"""
File Downloader Module

This module defines the `FileDownloader` class and its subclasses, which provide utilities
for downloading files from various sources, including HTTP, FTP, and Google Drive. It supports
features such as file locking, progress tracking, and handling incomplete downloads.

Classes:
    - FileDownloader: Abstract base class for file downloaders.
    - FTPFileDownloader: Downloads files from FTP servers.
    - HTTPDownloader: Downloads files from HTTP/HTTPS URLs.
    - GoogleDriveDownloader: Downloads files from Google Drive.

Dependencies:
    - os: For file path operations.
    - shutil: For moving files.
    - abc: For defining abstract base classes.
    - ftplib.FTP: For FTP operations.
    - typing: For type annotations.
    - urllib.parse: For parsing URLs.
    - gdown: For downloading files from Google Drive.
    - requests: For HTTP requests.
    - tqdm: For progress tracking.
    - filelock: For file locking.
    - atria_corelogger.logger: For logging utilities.
    - atria.data.datasets.downloads.download_file_info: For managing file download information.

Author: Your Name (your.email@example.com)
Date: 2025-04-07
Version: 1.0.0
License: MIT
"""

import shutil
from abc import ABC, abstractmethod
from urllib.parse import ParseResult

from atria_core.logger.logger import get_logger
from atria_core.utilities.repr import RepresentationMixin
from filelock import FileLock

from atria_datasets.core.download_manager.download_file_info import DownloadFileInfo

logger = get_logger(__name__)


class FileDownloader(ABC, RepresentationMixin):
    """
    Abstract base class for file downloaders.

    This class provides a common interface for downloading files from various sources.
    Subclasses must implement the `_download` method for specific protocols.

    Methods:
        - download: Downloads a file and handles locking and incomplete downloads.
        - from_url: Factory method to create a downloader based on the URL scheme.
    """

    def download(self, download_file_info: DownloadFileInfo) -> None:
        """
        Downloads a file and handles locking and incomplete downloads.

        Args:
            download_file_info (DownloadFileInfo): Information about the file to download.

        Returns:
            bool: True if the download is successful, False otherwise.

        Raises:
            RuntimeError: If the download fails.
        """
        lock_file_path = download_file_info.download_path.with_suffix(
            download_file_info.download_path.suffix + ".lock"
        )
        with FileLock(lock_file_path):
            if download_file_info.download_path.exists():
                logger.debug(f"{download_file_info.download_path} already exists.")
                return

            incomplete_destination_path = download_file_info.download_path.with_suffix(
                download_file_info.download_path.suffix + ".incomplete"
            )
            logger.debug(
                f"Downloading {download_file_info.url} to {incomplete_destination_path}"
            )
            try:
                self._download(
                    parsed_url=download_file_info.parsed_url,
                    destination_path=str(incomplete_destination_path),
                )
            except Exception as e:
                if incomplete_destination_path.exists():
                    incomplete_destination_path.unlink()
                raise RuntimeError(
                    f"Failed to download {download_file_info.url} to {incomplete_destination_path}"
                ) from e
            logger.debug(
                f"Download completed. Moving {incomplete_destination_path} to {download_file_info.download_path}"
            )
            shutil.move(incomplete_destination_path, download_file_info.download_path)

    @classmethod
    def from_url(cls, parsed_url: ParseResult, **kwargs) -> "FileDownloader":
        """
        Factory method to create a downloader based on the URL scheme.

        Args:
            parsed_url (ParseResult): The parsed URL of the file.
            **kwargs: Additional arguments for the downloader.

        Returns:
            FileDownloader: An instance of a subclass of `FileDownloader`.

        Raises:
            ValueError: If the URL scheme is unsupported.
        """
        if parsed_url.hostname == "drive.google.com":
            return GoogleDriveDownloader()
        elif parsed_url.scheme == "http" or parsed_url.scheme == "https":
            return HTTPDownloader(**kwargs)
        elif parsed_url.scheme == "ftp":
            return FTPFileDownloader(**kwargs)
        else:
            raise ValueError(f"Unsupported download URL: {parsed_url}")

    @abstractmethod
    def _download(self, parsed_url: ParseResult, destination_path: str) -> None:
        """
        Abstract method for downloading a file.

        Args:
            parsed_url (ParseResult): The parsed URL of the file.
            destination_path (str): The path where the file will be saved.

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.
        """
        raise NotImplementedError("Subclasses must implement this method.")


class FTPFileDownloader(FileDownloader):
    """
    Downloads files from FTP servers.

    Methods:
        - _download: Downloads a file from an FTP server.
    """

    def _download(self, parsed_url: ParseResult, destination_path: str) -> None:
        """
        Downloads a file from an FTP server.

        Args:
            parsed_url (ParseResult): The parsed URL of the file.
            destination_path (str): The path where the file will be saved.
        """
        from ftplib import FTP

        assert parsed_url.hostname is not None, "Parsed URL must have a hostname."
        ftp = FTP(parsed_url.hostname)
        ftp.login()
        with open(destination_path, "wb") as f:
            ftp.retrbinary(f"RETR {parsed_url.path.lstrip('/')}", f.write)
        ftp.quit()


class HTTPDownloader(FileDownloader):
    """
    Downloads files from HTTP/HTTPS URLs.

    Attributes:
        proxies (Optional[Dict]): Proxy settings for the HTTP request.
        user_agent (Optional[str]): The user agent string for the HTTP request.
        timeout (int): The timeout for the HTTP request.

    Methods:
        - _download: Downloads a file from an HTTP/HTTPS URL.
    """

    def __init__(
        self,
        proxies: dict | None = None,
        user_agent: str | None = None,
        timeout: int = 10,
    ):
        """
        Initializes the `HTTPDownloader`.

        Args:
            proxies (Optional[Dict]): Proxy settings for the HTTP request.
            user_agent (Optional[str]): The user agent string for the HTTP request.
            timeout (int): The timeout for the HTTP request.
        """
        self.proxies = proxies
        self.user_agent = user_agent
        self.timeout = timeout

    def _download(self, parsed_url: ParseResult, destination_path: str) -> None:
        """
        Downloads a file from an HTTP/HTTPS URL.

        Args:
            parsed_url (ParseResult): The parsed URL of the file.
            destination_path (str): The path where the file will be saved.

        Raises:
            Exception: If the HTTP request fails.
        """

        import requests

        try:
            from fsspec import url_to_fs
            from fsspec.callbacks import TqdmCallback

            try:
                logger.info(f"Downloading {parsed_url.geturl()} to {destination_path}")
                fs, path = url_to_fs(parsed_url.geturl())
                fs.get_file(path, destination_path, callback=TqdmCallback())
            except Exception as e:
                logger.error(f"Error downloading {parsed_url.geturl()}: {e}")
                raise
        except requests.RequestException as e:
            logger.error(f"Error downloading {parsed_url.geturl()}: {e}")
            raise


class GoogleDriveDownloader(FileDownloader):
    """
    Downloads files from Google Drive.

    Methods:
        - _download: Downloads a file from Google Drive.
    """

    def _download(self, parsed_url: ParseResult, destination_path: str) -> None:
        """
        Downloads a file from Google Drive.

        Args:
            parsed_url (ParseResult): The parsed URL of the file.
            destination_path (str): The path where the file will be saved.
        """

        import gdown

        file_id = parsed_url.path.split("/")[-2]
        gdown_url = f"{parsed_url.scheme}://{parsed_url.hostname}/uc?id={file_id}"
        gdown.download(gdown_url, str(destination_path), quiet=False)
