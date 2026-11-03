import pickle, io, zipfile, numpy as np
class RU(pickle.Unpickler):
    OK = {'builtins','collections','numpy','numpy.core.multiarray','numpy._core.multiarray','numpy.core.numeric','numpy._core.numeric','pandas','pandas.core.frame','pandas.core.series','pandas._libs.internals','pandas.core.internals.managers','pandas.core.indexes.base','pandas.core.indexes.range','pandas._libs.arrays','pandas.core.arrays.numpy_','pandas.core.internals.blocks','pandas.core.indexes.multi','pandas.core.arrays.string_','pandas.core.arrays.integer','pandas.core.arrays.floating'}
    def find_class(self, m, n):
        if m.split('.')[0] in ('numpy','pandas','collections','builtins') :
            if n in ('eval','exec','open','__import__','compile','getattr','system'): raise pickle.UnpicklingError(n)
            return super().find_class(m, n)
        raise pickle.UnpicklingError(f'blocked {m}.{n}')
def load(zpath, name):
    with zipfile.ZipFile(zpath) as z:
        return RU(io.BytesIO(z.read(name))).load()
