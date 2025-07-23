from dataclasses import dataclass

@dataclass
class FeatureKeyList:
    label: list[str]
    metadata: list[str]

    def __add__(self, other):
        return FeatureKeyList(
            label=self.label + other.label,
            metadata=self.metadata + other.metadata
        )
    def __len__(self):
        return len(self.label)
    
    def index(self, key: str) -> int:
        return self.label.index(key)
    

    def index_pair(self, label_key: str, metadata_key: str) -> int:
        """
        Returns the index where both label and metadata match the provided values.
        Raises ValueError if not found.
        """
        for idx, (lbl, meta) in enumerate(zip(self.label, self.metadata)):
            if lbl == label_key and meta == metadata_key:
                return idx
        raise ValueError(f"Pair ({label_key}, {metadata_key}) not found in FeatureKeyList.")
