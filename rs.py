import reedsolo
import configparser


def main():
    config = configparser.ConfigParser()
    config.read('settings/settings.ini')

    rs = reedsolo.RSCodec(config.getint("rs", "ecc"), config.getint("rs", "max_size"))

    message = 'Hello world'

    encoded_message = RSEncode(rs, message)

    print('Encoded message:', encoded_message)

    repaired_message = RSDecode(rs, encoded_message)

    print('Repaired message:', repaired_message)


def RSEncode(rs, command):
    return rs.encode(command.encode("utf-8"))


def RSDecode(rs, command):
    return rs.decode(command)[0].decode("utf-8")


if __name__ == '__main__':
    main()
