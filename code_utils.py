import configparser

import galois
import reedsolo


def create_coder(config):
    wrong_mode_raise(config)

    if config.get('mode', 'mode') == "bch":
        return galois.BCH(config.getint('bch', 'n'), config.getint('bch', 'k'))

    if config.get('mode', 'mode') == "rs":
        return reedsolo.RSCodec(config.getint("rs", "ecc"), config.getint("rs", "max_size"))


def encode(thingToEncode, coder, config):
    if config.get('mode', 'mode') == "bch":
        return coder.encode(thingToEncode).tolist()

    if config.get('mode', 'mode') == "rs":
        encoded_data = coder.encode(thingToEncode)
        res = []
        for i in encoded_data:
            res.append(i)

        return from_byte_to_bit(res, config)


def decode(pkg, coder, config):
    if config.get('mode', 'mode') == "bch":
        return coder.decode(pkg).tolist()

    if config.get('mode', 'mode') == "rs":
        decoded_data = coder.decode(from_bit_to_byte(pkg, config))[0]
        res = []
        for i in decoded_data:
            res.append(i)

        return res


def wrong_mode_raise(config):
    if config.get('mode', 'mode') != "rs" and config.get('mode', 'mode') != "rs":
        raise Exception("Wrong parameter for coder")


def from_byte_to_bit(byte_pkg, config):
    end_of_original_codeword = config.getint("rs", "max_size") - config.getint("rs", "ecc") * 8

    res_pkg = byte_pkg[:end_of_original_codeword]
    for number in byte_pkg[end_of_original_codeword:]:
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


def from_bit_to_byte(start_pkg, config):
    end_of_original_codeword = config.getint("rs", "max_size") - config.getint("rs", "ecc") * 8

    res_pkg = start_pkg[:end_of_original_codeword]
    blocker = end_of_original_codeword
    while blocker < config.getint("rs", "max_size"):
        res_pkg.append(to_dex(start_pkg[blocker:blocker + 8]))
        blocker += 8

    return res_pkg


def to_dex(list_of_binary):
    res = list_of_binary[0] * 128 + list_of_binary[1] * 64 + list_of_binary[2] * 32 + list_of_binary[3] * 16 + \
          list_of_binary[4] * 8 + list_of_binary[5] * 4 + list_of_binary[6] * 2 + list_of_binary[7] * 1
    return res


if __name__ == '__main__':

    config = configparser.ConfigParser()
    config.read('settings/settings.ini')

    coder = create_coder(config)
    # coder = galois.ReedSolomon(255, 231)

    mass = [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0,
            0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1,
            0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1,
            0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0,
            0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1,
            0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1,
            1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0]

    print(mass)

    encoded_mass = encode(mass, coder, config)
    # encoded_mass = coder.encode(mass).tolist()

    print(encoded_mass)
    print(len(encoded_mass))

    val = decode(encoded_mass, coder, config)
    # val = coder.decode(encoded_mass).tolist()

    print(val)
    print(len(val))

    if mass == val:
        print("IT'S DUCKING WORK!!")
