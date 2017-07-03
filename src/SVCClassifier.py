class SVCClassifier:
    
    svc = None
    scaler = None
    
    def __init__(self,svc,scaler):
        self.svc = svc
        self.scaler = scaler

    def predict(self, features):
        scaled_features = self.scaler.transform(features)  
        return self.svc.predict(scaled_features)
