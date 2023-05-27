import bchlib

if __name__ == '__main__':

    for poly in range(150, 3000):

        for t in range(1, 63):
            try:
                a = bchlib.BCH(poly, t)
            except RuntimeError:
                a = None
                print(f"{poly} and {t} not working")

            if a is not None:
                with open("numberOfRun.txt", "a") as file:
                    file.write(f"For ({a.n}, {a.t}) we need poly = {poly} and t = {t}\n")

