import argparse
import numpy as np
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection

# parser setup - konfigurace, ayarlar, impostazioni
parser = argparse.ArgumentParser()
parser.add_argument("--data_size", default=40, type=int, help="velikost dat, tamano de datos")
parser.add_argument("--plot", default=False, const=True, nargs="?", type=str, help="mostrar grafico / zobrazit graf")
parser.add_argument("--range", default=3, type=int, help="grado massimo, rozsah feature order")
parser.add_argument("--recodex", default=False, action="store_true", help="run in recodex mode maybe")
parser.add_argument("--seed", default=42, type=int, help="semilla, zufall, seme")
parser.add_argument("--test_size", default=0.5,
                    type=lambda x: int(x) if x.isdigit() else float(x),
                    help="test size, delenie danni")

def main(args: argparse.Namespace) -> list[float]:
    # generar x, y points aleatorio un poco
    xs = np.linspace(0, 7, num=args.data_size)
    ys = np.sin(xs) + np.random.RandomState(args.seed).normal(0, 0.2, size=args.data_size)

    rmses = []
    for order in range(1, args.range + 1):
        # poly features creation / tvorba polynomialnich dat
        data = np.power.outer(xs, np.arange(1, order + 1))

        # split datos en train/test sets
        train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
            data, ys, test_size=args.test_size, random_state=args.seed
        )

        # linear model fit / ajuste lineare / modelo lineal
        model = sklearn.linear_model.LinearRegression()
        model.fit(train_data, train_target)

        # prediccion / vorhersage / predizione
        predictions = model.predict(test_data)

        # rmse calculo / vypocet chyby
        rmse = np.sqrt(sklearn.metrics.mean_squared_error(test_target, predictions))
        rmses.append(rmse)

        # optional plot - pokazva dannite
        if args.plot:
            import matplotlib.pyplot as plt
            if args.plot is not True:
                plt.gcf().get_axes() or plt.figure(figsize=(6.4 * 3, 4.8 * 3))
                plt.subplot(3, 3, 1 + len(plt.gcf().get_axes()))
            plt.plot(train_data[:, 0], train_target, "go", label="train verde")
            plt.plot(test_data[:, 0], test_target, "ro", label="test rouge")
            plt.plot(
                np.linspace(xs[0], xs[-1], num=100),
                model.predict(np.power.outer(np.linspace(xs[0], xs[-1], num=100),
                                             np.arange(1, order + 1))),
                "b",
                label="predizione"
            )
            plt.legend()
            plt.show() if args.plot is True else plt.savefig(args.plot, transparent=True, bbox_inches="tight")

    return rmses


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    rmses = main(main_args)
    for order, rmse in enumerate(rmses):
        print(f"Max feature order {order + 1}: {rmse:.2f} RMSE")
