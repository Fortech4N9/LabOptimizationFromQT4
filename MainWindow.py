# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'MainWindow.ui'
#
# Created by: PyQt5 UI code generator 5.15.7
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(753, 511)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/newPrefix/res/icon.ico"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        MainWindow.setWindowIcon(icon)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.zInput = QtWidgets.QDoubleSpinBox(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.zInput.setFont(font)
        self.zInput.setProperty("value", 10.0)
        self.zInput.setObjectName("zInput")
        self.gridLayout.addWidget(self.zInput, 1, 3, 1, 1)
        self.rInput = QtWidgets.QDoubleSpinBox(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.rInput.setFont(font)
        self.rInput.setProperty("value", 0.1)
        self.rInput.setObjectName("rInput")
        self.gridLayout.addWidget(self.rInput, 2, 3, 1, 1)
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 1, 2, 1, 1)
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setObjectName("label_4")
        self.gridLayout.addWidget(self.label_4, 0, 10, 1, 1)
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(11)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        self.gridLayout.addWidget(self.label_5, 0, 2, 1, 1)
        self.xInput = QtWidgets.QLineEdit(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.xInput.setFont(font)
        self.xInput.setInputMask("")
        self.xInput.setObjectName("xInput")
        self.gridLayout.addWidget(self.xInput, 1, 11, 1, 1)
        self.epsInput = QtWidgets.QDoubleSpinBox(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.epsInput.setFont(font)
        self.epsInput.setDecimals(4)
        self.epsInput.setSingleStep(0.005)
        self.epsInput.setProperty("value", 0.01)
        self.epsInput.setObjectName("epsInput")
        self.gridLayout.addWidget(self.epsInput, 0, 11, 1, 1)
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 1, 10, 1, 1)
        self.funcSelector = QtWidgets.QComboBox(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.funcSelector.setFont(font)
        self.funcSelector.setObjectName("funcSelector")
        self.funcSelector.addItem("")
        self.funcSelector.addItem("")
        self.funcSelector.addItem("")
        self.funcSelector.addItem("")
        self.funcSelector.addItem("")
        self.funcSelector.addItem("")
        self.funcSelector.addItem("")
        self.funcSelector.addItem("")
        self.gridLayout.addWidget(self.funcSelector, 0, 3, 1, 1)
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 2, 2, 1, 1)
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_6.setFont(font)
        self.label_6.setObjectName("label_6")
        self.gridLayout.addWidget(self.label_6, 8, 2, 1, 1)
        self.calculateBtn = QtWidgets.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(11)
        self.calculateBtn.setFont(font)
        self.calculateBtn.setObjectName("calculateBtn")
        self.gridLayout.addWidget(self.calculateBtn, 8, 11, 1, 1)
        self.resOutput = QtWidgets.QLineEdit(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.resOutput.setFont(font)
        self.resOutput.setReadOnly(True)
        self.resOutput.setObjectName("resOutput")
        self.gridLayout.addWidget(self.resOutput, 8, 3, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "WindowTitle"))
        self.label_3.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:14pt;\">z</span></p></body></html>"))
        self.label_4.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:12pt;\">eps</span></p></body></html>"))
        self.label_5.setText(_translate("MainWindow", "<html><head/><body><p>?????????? ??????????????</p></body></html>"))
        self.xInput.setText(_translate("MainWindow", "0; 0"))
        self.label.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:12pt;\">(x1; x2)</span></p></body></html>"))
        self.funcSelector.setItemText(0, _translate("MainWindow", "1.1"))
        self.funcSelector.setItemText(1, _translate("MainWindow", "1.2"))
        self.funcSelector.setItemText(2, _translate("MainWindow", "1.3"))
        self.funcSelector.setItemText(3, _translate("MainWindow", "2.4"))
        self.funcSelector.setItemText(4, _translate("MainWindow", "2.8"))
        self.funcSelector.setItemText(5, _translate("MainWindow", "2.12"))
        self.funcSelector.setItemText(6, _translate("MainWindow", "2.16"))
        self.funcSelector.setItemText(7, _translate("MainWindow", "?????????????? ??????????????????"))
        self.label_2.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:14pt;\">r</span><span style=\" font-size:14pt; vertical-align:super;\">0</span></p></body></html>"))
        self.label_6.setText(_translate("MainWindow", "??????????"))
        self.calculateBtn.setText(_translate("MainWindow", "?????????????????? ??????????????"))
import res_rc
