import bchlib
import reedsolo


def create_coder(config):
    wrong_mode_raise(config)

    if config.get('mode', 'mode') == "bch":
        return bchlib.BCH(config.getint('bch', 'poly'), config.getint('bch', 't'))

    if config.get('mode', 'mode') == "rs":
        return reedsolo.RSCodec(config.getint("rs", "ecc"), config.getint("rs", "max_size"))


def encode(thingToEncode, coder, config):
    if config.get('mode', 'mode') == "bch":
        data = str.encode(thingToEncode, encoding="utf-8")
        ecc = coder.encode(data)
        pkg = data + ecc
        return pkg

    if config.get('mode', 'mode') == "rs":
        encoded_data = coder.encode(thingToEncode.encode("utf-8"))
        res = []
        for i in encoded_data:
            res.append(i)

        return res


def decode(pkg, coder, config):
    if config.get('mode', 'mode') == "bch":
        data, ecc = pkg[:-coder.ecc_bytes], pkg[-coder.ecc_bytes:]
        val = coder.decode(data, ecc)
        return val[1].decode("utf-8")

    if config.get('mode', 'mode') == "rs":
        decoded_data = coder.decode(pkg)[0].decode("utf-8")
        res = []
        for i in decoded_data:
            res.append(i)

        return res


def wrong_mode_raise(config):
    if config.get('mode', 'mode') != "rs" and config.get('mode', 'mode') != "rs":
        raise Exception("Wrong parameter for coder")


def from_byte_to_bit(byte_pkg):

    start_pkg = []
    for i in byte_pkg:
        start_pkg.append(i)

    res_pkg = start_pkg[:231]
    for number in start_pkg[231:]:
        res_pkg.extend(to_binary(number))

    return res_pkg


def to_binary(number):
    check = [128, 64, 32, 16, 8, 4, 2, 1]
    res = []

    for i in check:
        if number >= i:
            number = number - i
            res.append(1)
        else:
            res.append(0)

    return res


def from_bit_to_byte(start_pkg):
    res_pkg = start_pkg[:231]
    res_pkg.append(to_dex(start_pkg[231:239]))
    res_pkg.append(to_dex(start_pkg[239:247]))
    res_pkg.append(to_dex(start_pkg[247:]))
    return res_pkg


def to_dex(list_of_binary):
    res = list_of_binary[0] * 128 + list_of_binary[1] * 64 + list_of_binary[2] * 32 + list_of_binary[3] * 16 + \
          list_of_binary[4] * 8 + list_of_binary[5] * 4 + list_of_binary[6] * 2 + list_of_binary[7] * 1
    return res


if __name__ == '__main__':
    coder = bchlib.BCH(256, 3)

    mass = [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0,
            0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1,
            0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1,
            0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0,
            0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1,
            0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1,
            1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0]

    # ecc = coder.encode(b'hello')
    # pkg = b'hello' + ecc
    # print(pkg)
    #
    # data, ecc = pkg[:-coder.ecc_bytes], pkg[-coder.ecc_bytes:]
    # val = coder.decode(data, ecc)
    # print(val[1])

    # print(str(mass)[1:len(str(mass)) - 1].replace(", ", ""))

    ecc = coder.encode(bytes(mass))
    pkg = bytes(mass) + ecc
    print(pkg)

    binary_pkg = from_byte_to_bit(pkg)

    byte_pkg = bytes(from_bit_to_byte(binary_pkg))
    data, ecc = byte_pkg[:-coder.ecc_bytes], byte_pkg[-coder.ecc_bytes:]

    val = coder.decode(data, ecc)

    print(val[1])
