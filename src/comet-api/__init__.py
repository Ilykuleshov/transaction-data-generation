from comet_ml.api import API


class ExportComets:

    def __init__(self, api_token: str) -> None:
        self.api = API(api_token)
    
    
