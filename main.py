import sys  # sys нужен для передачи argv в QApplication
import os  # Отсюда нам понадобятся методы для отображения содержимого директорий

from PyQt5 import QtWidgets
from scipy import optimize
import numpy as np
from typing import Callable, List

from PySide6.QtCore import Slot
import MainWindow  # Это наш конвертированный файл дизайна

class ExampleApp(QtWidgets.QMainWindow, MainWindow.Ui_MainWindow):
    def __init__(self):
        # Это здесь нужно для доступа к переменным, методам
        # и т.д. в файле design.py
        super().__init__()
        self.setupUi(self)  # Это нужно для инициализации нашего дизайна
        self.calculateBtn.clicked.connect(self.onCalculateClicked)
        self.show()

    @Slot()
    def onCalculateClicked(self):
        funcNum = self.funcSelector.currentIndex()
        x = [float(numeric_string) for numeric_string in self.xInput.text().split("; ")]
        eps = self.epsInput.value()
        r0 = self.rInput.value()
        z = self.zInput.value()
        res = penalty(x, f1, r0, z, eps, restrictionsOfEquality[funcNum], restrictionsOfNotEquality[funcNum])

        r = "".join(f"{j:.{4}f}, " for j in res)[:-2]

        self.resOutput.setText(f"f1({r}) = {f1(res):.{4}f}")

    # Метод штрафных функций


def getPenaltyAuxilitaryFunctionResult(f, r, rest_eq, rest_not_eq, x):
    x1 = x[0]
    x2 = x[1]
    H = 0
    for i in rest_eq:
        H += pow(abs(i(x1, x2)), 2)
    for i in rest_not_eq:
        H += pow(max(0, i(x1, x2)), 2)
    return f(x) + r * H


def penalty(x0, f, r, z, eps, rest_eq, rest_not_eq):
    xcur = np.array(x0)
    xnew = optimal_gradient_method(lambda x: getPenaltyAuxilitaryFunctionResult(f, r, rest_eq, rest_not_eq, x), xcur,
                                   eps)
    while ((xcur - xnew) ** 2).sum() > eps:
        r *= z
        xcur = xnew
        xnew = optimal_gradient_method(lambda x: getPenaltyAuxilitaryFunctionResult(f, r, rest_eq, rest_not_eq, x),
                                       xcur, eps)
    return xnew


# Метод барьерных функций
def getBarrierAuxilitaryFunctionResult(f, r, rest_not_eq, x):
    x1, x2 = x
    H = sum(1 / (0.000000001 + pow(max(0, -i(x1, x2)), 2)) for i in rest_not_eq)
    return f(x) + r * H


def barrier(x0, f, r, z, eps, rest_not_eq):
    xcur = np.array(x0)
    xnew = None
    atLeastOnePointFound = False
    while not (atLeastOnePointFound and (((xcur - xnew) ** 2).sum() < eps ** 2)):
        xtemp = optimal_gradient_method(lambda x: getBarrierAuxilitaryFunctionResult(f, r, rest_not_eq, x), xcur)

        isInside = not any(neq(xtemp[0], xtemp[1]) > eps for neq in rest_not_eq)

        if (isInside):
            if not atLeastOnePointFound:
                atLeastOnePointFound = True
            else:
                xcur = xnew
            xnew = xtemp

        r *= z

    return xnew


# Оптимальный градиентный спуск
def euclidean_norm(h: np.array):
    return np.sqrt((h ** 2).sum())


def optimal_gradient_method(func: Callable[[List[float]], float], x0: List[float], eps: float = 0.001):
    x = np.array(x0)

    def grad(func, xcur, eps) -> np.array:
        return optimize.approx_fprime(xcur, func, eps ** 2)

    gr = grad(func, x, eps)
    a = 0.

    while any([abs(gr[i]) > eps for i in range(len(gr))]):
        # while euclidean_norm(gr) > eps:
        gr = grad(func, x, eps)
        a = optimize.minimize_scalar(lambda koef: func(*[x + koef * gr])).x
        x += a * gr
        if a == 0:
            break

    return x


# Функции
def f1(x):
    x1, x2 = x
    return x1 ** 2 + x2 ** 2


def f22(x):
    x1, x2 = x
    return x1 + x2



restrictionsOfEquality = [
    [lambda x1, x2: x1 + x2 - 2],
    [lambda x1, x2: x1 - 1],
    [],
    []
]

restrictionsOfNotEquality = [
    [],
    [lambda x1,x2: x1+x2-2],
    [lambda x1,x2: x1+x2-2, lambda x1,x2: -x1+1],
    [lambda x1,x2: x1*x1-x2, lambda x1,x2: -x1]
]
def main():
    app = QtWidgets.QApplication(sys.argv)  # Новый экземпляр QApplication
    window = ExampleApp()  # Создаём объект класса ExampleApp
    window.show()  # Показываем окно
    app.exec_()  # и запускаем приложение

if __name__ == '__main__':  # Если мы запускаем файл напрямую, а не импортируем
    main()  # то запускаем функцию main()


