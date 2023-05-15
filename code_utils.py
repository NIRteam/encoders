import bchlib
import reedsolo


def create_coder(config):
    wrong_mode_raise(config)
    
    if config.get('mode', 'mode') == "bch":
        return bchlib.BCH(config.getint('bch', 'poly'), config.getint('bch', 't'))

    if config.get('mode', 'mode') == "rs":
        return reedsolo.RSCodec(config.getint("rs", "ecc"), config.getint("rs", "max_size"))


def encode(command, coder, config):
    if config.get('mode', 'mode') == "bch":
        data = str.encode(command, encoding="utf-8")
        ecc = coder.encode(data)
        pkg = data + ecc
        return pkg

    if config.get('mode', 'mode') == "rs":
        return coder.encode(command.encode("utf-8"))


def decode(pkg, coder, config):
    if config.get('mode', 'mode') == "bch":
        data, ecc = pkg[:-coder.ecc_bytes], pkg[-coder.ecc_bytes:]
        val = coder.decode(data, ecc)
        return val[1].decode("utf-8")

    if config.get('mode', 'mode') == "rs":
        return coder.decode(pkg)[0].decode("utf-8")


def wrong_mode_raise(config):
    if config.get('mode', 'mode') != "rs" and config.get('mode', 'mode') != "rs":
        raise Exception("Wrong parameter for coder")
