import re
from dataclasses import dataclass


@dataclass
class FeatureKeyList:
    label: list[str]
    metainfo: list[str]

    def __post_init__(self):
        assert len(self.label) == len(self.metainfo), "label/metainfo lengths must match"

    def __add__(self, other):
        return FeatureKeyList(
            label=self.label + other.label,
            metainfo=self.metainfo + other.metainfo,
        )

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return FeatureKeyList(
                label=self.label[idx],
                metainfo=self.metainfo[idx],
            )
        if hasattr(idx, "__iter__"):
            return FeatureKeyList(
                label=[self.label[i] for i in idx],
                metainfo=[self.metainfo[i] for i in idx],
            )
        return FeatureKeyList(
            label=[self.label[idx]],
            metainfo=[self.metainfo[idx]],
        )

    def __eq__(self, other):
        if not hasattr(other, "label") or not hasattr(other, "metainfo"):
            return False
        return self.label == other.label and self.metainfo == other.metainfo

    def __repr__(self):
        if not self.label:
            return "empty"

        out = ""
        prev_prefix = None
        prev_metainfo = None
        cnt = 1
        prefix = self.label[0]
        metainfo = self.metainfo[0]

        for label, metainfo in zip(self.label, self.metainfo):
            res = re.match(r"^.+_\d+", label)
            prefix = label if res is None else label.rsplit("_", 1)[0]

            if prev_prefix and prev_prefix == prefix and prev_metainfo == metainfo:
                cnt += 1
            elif prev_prefix and prev_metainfo:
                out += f"{prev_prefix} ({prev_metainfo}) x{cnt}, "
                cnt = 1

            prev_prefix = prefix
            prev_metainfo = metainfo

        out += f"{prefix} ({metainfo}) x{cnt}"
        return out

    def index(self, key: str) -> int:
        return self.label.index(key)

    def index_pair(self, label_key: str, metainfo_key: str) -> int:
        for idx, (label, metainfo) in enumerate(zip(self.label, self.metainfo)):
            if label == label_key and metainfo == metainfo_key:
                return idx
        raise ValueError(f"Pair ({label_key}, {metainfo_key}) not found in FeatureKeyList.")

    def index_metainfo(self, key: str):
        return [i for i, metainfo in enumerate(self.metainfo) if metainfo == key]

    def filter_metainfo(self, key: str):
        return self[self.index_metainfo(key)]

    def has_label(self, key: str) -> bool:
        return key in self.label

    def dump(self):
        return [f"{label}, {metainfo}" for label, metainfo in zip(self.label, self.metainfo)]

    @staticmethod
    def load(input_keys):
        if not input_keys:
            return FeatureKeyList([], [])
        labels, metainfos = zip(*[s.split(", ", 1) for s in input_keys])
        return FeatureKeyList(label=list(labels), metainfo=list(metainfos))
