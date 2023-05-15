import bchlib


def create_coder(config):
    if config.get('mode', 'mode') == "bch":
        return bchlib.BCH(config.getint('bch', 'poly'), config.getint('bch', 't'))


def encode(command, coder, config):
    if config.get('mode', 'mode') == "bch":
        data = str.encode(command, encoding="utf-8")
        ecc = coder.encode(data)
        pkg = data + ecc
        return pkg


def decode(pkg, coder, config):
    if config.get('mode', 'mode') == "bch":
        data, ecc = pkg[:-coder.ecc_bytes], pkg[-coder.ecc_bytes:]
        val = coder.decode(data, ecc)
        return val[1].decode("utf-8")
