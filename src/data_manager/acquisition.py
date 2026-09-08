"""Complete active requests independently of retained cache capacity."""

from download_unit import DataRequest
from download_unit.data_assembly import assembleData


class MarketDataAcquisition:
    """Own coverage lookup, retained fragments, downloads and result assembly."""

    def __init__(self, dataManager, downloadManager) -> None:
        self.dataManager = dataManager
        self.downloadManager = downloadManager

    def acquire(self, request: DataRequest):
        request = request.withChanges(datasetIdentity=self.downloadManager.datasetIdentity(request))
        complete = self.dataManager.getData(request)
        if complete is not None:
            return complete
        missing = self.dataManager.getMissingRequests(request)
        available = self.dataManager.getAvailableData(request)
        fragments = [] if available is None else [available]
        for fragment in missing:
            downloaded = self.downloadManager.downloadDataType(fragment.data_type, fragment)
            fragments.append(self.dataManager.storeData(fragment, downloaded))
        return assembleData(request, fragments)
