import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.3
import QtQuick.Dialogs

ApplicationWindow  {
    id: mainWindow
    width: 490
    height: 300
    minimumWidth: 490
    minimumHeight: 310
    maximumHeight: 310
    visible: true
    title: "Image verificator"
    readonly property real szNormalMargin: 16
    color: "silver"

    Connections {
        target: predicter
        function onImagePredicted(prediction) {
            textEdit.text += prediction
        }

        function onTrainingProcessChanged() {
            if (!predicter.trainingProcess) {
                dialogLearningCompleted.open()
                mainRect.learningCompleted = true
            }
        }
    }
//    background: Rectangle {
//        color: "silver"
//    }
    Connections {
        target: signalHelper
        function onMessageSignal(message) {
            statisticsTextEdit.text += message
        }
    }

    Rectangle {
        id: mainRect
        anchors.fill: parent
        color: "silver"
        anchors.leftMargin: szNormalMargin
        anchors.topMargin: szNormalMargin
        anchors.rightMargin: szNormalMargin
        property bool learningCompleted: false

        RowLayout {
            id: verificationLayout
            anchors.left: parent.left
            anchors.right: parent.right
            anchors.top: parent.top
            //width: 300
            spacing: 0

            ColumnLayout {
                Text {
                    text: "Проверка изображения на подлинность"
                    font.pixelSize: 16
                }

                Flickable {
                    Layout.fillWidth: true
                    Layout.preferredHeight: 200
                    flickableDirection: Flickable.VerticalFlick
                    TextArea.flickable: TextArea {
                        id: textEdit
                        wrapMode: TextEdit.Wrap
                        readOnly: true
                        onTextChanged: {
                            cursorPosition = length-1
                        }
                    }
                    ScrollBar.vertical: ScrollBar {

                    }
                }
                RowLayout {
                    Layout.fillWidth: true

                    Button {
                        id: btnTrainModel
                        Layout.preferredHeight: 50
                        Layout.fillWidth: true
                        contentItem: Text {
                            text: "Обучить модель"
                            wrapMode: Text.Wrap
                            verticalAlignment: Text.AlignVCenter
                            horizontalAlignment: Text.AlignHCenter
                        }
                        visible: !predicter.trainingProcess
                        onClicked: {
                            trainingPopup.open()
                        }
                    }

                    Button {
                        id: btnShowStatistics
                        Layout.preferredHeight: 50
                        Layout.fillWidth: true
                        contentItem: Text {
                            text: "Статистика обучения"
                            wrapMode: Text.Wrap
                            verticalAlignment: Text.AlignVCenter
                            horizontalAlignment: Text.AlignHCenter
                        }
                        visible: predicter.trainingProcess || mainRect.learningCompleted
                        onClicked: {
                            statisticsPopup.open()
                        }
                    }

                    Button {
                        Layout.preferredHeight: 50
                        Layout.fillWidth: true
                        //Layout.preferredWidth: 80
                        //text: "Загрузить модель"
                        enabled: !predicter.trainingProcess
                        contentItem: Text {
                            text: "Загрузить модель"
                            wrapMode: Text.Wrap
                            verticalAlignment: Text.AlignVCenter
                            horizontalAlignment: Text.AlignHCenter
                        }
                        onClicked: {
                            loadModelDialog.open()
                        }
                    }
                    Button {
                        Layout.preferredHeight: 50
                        Layout.fillWidth: true
                        //Layout.preferredWidth: 80
                        //text: "Загрузить модель"
                        enabled: !predicter.trainingProcess && predicter.modelPrepared
                        contentItem: Text {
                            text: "Сохранить модель"
                            wrapMode: Text.Wrap
                            verticalAlignment: Text.AlignVCenter
                            horizontalAlignment: Text.AlignHCenter
                        }
                        onClicked: {
                            saveModelDialog.open()
                        }
                    }
                    Button {
                        //implicitWidth: 50
                        Layout.preferredHeight: 50
                        Layout.fillWidth: true
                        //Layout.maximumWidth: 150
                        contentItem: Text {
                            text: "Выбрать изображение для проверки"
                            wrapMode: Text.Wrap
                            verticalAlignment: Text.AlignVCenter
                            horizontalAlignment: Text.AlignHCenter
                        }

                        enabled: !predicter.trainingProcess && predicter.modelPrepared
                        onClicked: {
                            imagePredictDialog.open()
                            //predicter.predictImage("C:\\Users\\imynn\\Downloads\\CASIA2\\Au\\Au_ani_101899.jpg")
                        }
                    }
                }
                }
            }

        Dialog {
            id: dialogLearningCompleted
            anchors.centerIn: parent
            standardButtons: Dialog.Ok
            header: Text {
                text: "Результат обучения"
                horizontalAlignment: Text.AlignHCenter
            }

            Text {
                text: "Обучение модели завершено"
                wrapMode: Text.Wrap
            }
        }

        Popup {
            id: statisticsPopup
            anchors.centerIn: parent
            bottomMargin: szNormalMargin
            modal: true
            implicitWidth: parent.width
            Overlay.modal: Rectangle {
                color: "#aacfdbe7"
            }
            ColumnLayout {
                anchors.left: parent.left
                anchors.right: parent.right
                Text {
                    text: "Статистика обучения"
                    font.pixelSize: 16
                }

                Flickable {
                    Layout.fillWidth: true
                    Layout.preferredHeight: 230
                    flickableDirection: Flickable.VerticalFlick
                    TextArea.flickable: TextArea {
                        id: statisticsTextEdit
                        wrapMode: TextEdit.Wrap
                        readOnly: true
                        onTextChanged: {
                            cursorPosition = length-1
                        }
                    }
                    ScrollBar.vertical: ScrollBar {

                    }
                }
            }

        }

        Popup {
            id: trainingPopup
            anchors.centerIn: parent
            modal: true

            implicitWidth: 300
            implicitHeight: 200 + szNormalMargin
            Overlay.modal: Rectangle {
                color: "#aacfdbe7"
            }

            ColumnLayout {
                id: zz

                Text {
                    text: "Укажите путь до подлинных изображений"
                }

                RowLayout {
                    Layout.fillWidth: true

                    TextField {
                        Layout.fillWidth: true
                        readOnly: true

                        Text {
                            anchors.fill: parent
                            text: predicter.authenticPath
                            elide: Text.ElideRight
                        }
                    }

                    Button {
                        text: "Выбрать папку"
                        enabled: !predicter.trainingProcess
                        onClicked: {
                            authenticFolderDialog.open()
                        }
                    }
                }

                Text {
                    text: "Укажите путь до поддельных изображений"
                }

                RowLayout {
                    Layout.fillWidth: true

                    TextField {
                        Layout.fillWidth: true
                        readOnly: true

                        Text {
                            anchors.fill: parent
                            text: predicter.fakePath
                            elide: Text.ElideRight
                        }
                    }

                    Button {
                        text: "Выбрать папку"
                        enabled: !predicter.trainingProcess
                        onClicked: {
                            fakeFolderDialog.open()
                        }
                    }
                }
                RowLayout {
                    Layout.fillWidth: true
                    ColumnLayout {
                        Text {
                            text: "Количество эпох"
                        }

                        TextField {
                            id: txtFieldNumEpochs
                            validator: IntValidator {}
                            Component.onCompleted: {
                                text = predicter.epochs
                            }
                        }

                        Text {
                            text: "Число батчей"
                        }

                        TextField {
                            id: txtFieldBatchSize
                            validator: IntValidator {}
                            Component.onCompleted: {
                                text = predicter.batchSize
                            }
                        }
                    }

                    Button {
                        Layout.topMargin: szNormalMargin
                        Layout.alignment: Qt.AlignVCenter
                        Layout.preferredHeight: 50
                        text: "Начать обучение модели"
                        enabled: !predicter.trainingProcess && predicter.fakePath !== "" && predicter.authenticPath !== ""
                                 && predicter.fakePath !== predicter.authenticPath && txtFieldNumEpochs.text !== "" && txtFieldBatchSize.text !== ""
                        onClicked: {
                            predicter.epochs = txtFieldNumEpochs.text
                            predicter.batchSize = txtFieldBatchSize.text
                            predicter.runTraining()
                            trainingPopup.close()
                        }
                    }
                }
            }
        }

        FileDialog {
            id: loadModelDialog
            nameFilters: ["Saved model (*.pb)"]
            onAccepted: {
                predicter.loadModel(selectedFile.toString())
               // predictImage(selectedFile)
            }
        }

        FileDialog {
            id: saveModelDialog
            fileMode: FileDialog.SaveFile
            onAccepted: {
                predicter.saveModel(selectedFile.toString())
            }
        }

        FolderDialog {
            id: authenticFolderDialog
            onAccepted: {
                predicter.authenticPath = selectedFolder.toString()
            }
        }

        FolderDialog {
            id: fakeFolderDialog
            onAccepted: {
                predicter.fakePath = selectedFolder.toString()
            }
        }

        FileDialog {
            id: imagePredictDialog
            nameFilters: ["Images (*.jpg *.bmp *.png *.tif)"]
            onAccepted: {
                predicter.imagePath = selectedFile.toString()
                predicter.predictImage(selectedFile)
               // predictImage(selectedFile)
            }
        }
    }
}
