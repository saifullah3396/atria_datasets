def test_imports():
    from atria_datasets.datasets import (
        AtriaDataset,  # noqa
        AtriaHuggingfaceDataset,  # noqa
        DeltalakeReader,  # noqa
        DeltalakeStorageManager,  # noqa
        DownloadFileInfo,  # noqa
        DownloadManager,  # noqa
        FileDownloader,  # noqa
        FTPFileDownloader,  # noqa
        GoogleDriveDownloader,  # noqa
        HTTPDownloader,  # noqa
        MsgpackFileWriter,  # noqa
        MsgpackShardWriter,  # noqa
        ShardedDatasetStorageManager,  # noqa
        SplitIterator,  # noqa
        StandardSplitter,  # noqa
    )
