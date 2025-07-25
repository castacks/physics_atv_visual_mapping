import re
from dataclasses import dataclass

@dataclass
class FeatureKeyList:
    label: list[str]
    metainfo: list[str]

    def __add__(self, other):
        return FeatureKeyList(
            label=self.label + other.label,
            metainfo=self.metainfo + other.metainfo
        )
    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return FeatureKeyList(
                label=self.label[idx],
                metainfo=self.metainfo[idx]
            )
        elif hasattr(idx, '__iter__'):
            return FeatureKeyList(
                label=[self.label[i] for i in idx],
                metainfo=[self.metainfo[i] for i in idx]
            )
        else:
            return FeatureKeyList(
                label=[self.label[idx]],
                metainfo=[self.metainfo[idx]]
            )

    def __eq__(self, other):
        if len(self.label) != len(other.label):
            return False

        for x,y in zip(self.label, other.label):
            if x != y:
                return False

        return True

    def __repr__(self):
        """
        Make a simpler print that condenses similar keys
        """
        out = ""
        prev_prefix = None
        prev_metainfo = None
        cnt = 1
        for l, m in zip(self.label, self.metainfo):
            res = re.match(r"^.+_\d+", l)

            if res is None:
                prefix = l
            else:
                prefix = l.rsplit('_', 1)[0]

            if prev_prefix and prev_prefix == prefix and prev_metainfo and prev_metainfo == m:
                cnt += 1

            elif prev_prefix and prev_metainfo:
                out += f"{prev_prefix} ({prev_metainfo}) x{cnt}, "
                cnt = 1

            prev_prefix = prefix
            prev_metainfo = m

        out += f"{prefix} ({m}) x{cnt}"
        return out
    
    def index(self, key: str) -> int:
        return self.label.index(key)

    def index_pair(self, label_key: str, metainfo_key: str) -> int:
        """
        Returns the index where both label and metainfo match the provided values.
        Raises ValueError if not found.
        """
        for idx, (lbl, meta) in enumerate(zip(self.label, self.metainfo)):
            if lbl == label_key and meta == metainfo_key:
                return idx
        raise ValueError(f"Pair ({label_key}, {metainfo_key}) not found in FeatureKeyList.")

    def filter_metainfo(self, k):
        """
        Return a FeatureKeyList containing fks from self that have metainfo=k
        """
        idxs = [i for i,m in enumerate(self.metainfo) if m==k]
        return self[idxs]