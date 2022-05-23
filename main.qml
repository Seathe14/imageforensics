import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.3
import QtQuick.Dialogs

ApplicationWindow  {
    width: 400
    height: 300
    minimumWidth: 400
    minimumHeight: 200
    visible: true
    readonly property real szNormalMargin: 16

    Connections {
        target: predicter
        function onImagePredicted(prediction) {
            textEdit.text += prediction
        }
    }
    background: Rectangle {
        color: "silver"
    }

    Rectangle {
        anchors.fill: parent
        color: "silver"
        anchors.leftMargin: szNormalMargin
        anchors.topMargin: szNormalMargin
        anchors.rightMargin: szNormalMargin

        RowLayout {
            id: verificationLayout
            anchors.left: parent.left
            anchors.top: parent.top
            width: 300
            spacing: 0

            ColumnLayout {
                Text {
                    text: "Проверка изображения на подлинность"
                }

                Flickable {
                    Layout.fillWidth: true
                    Layout.preferredHeight: 150
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
                        Layout.preferredWidth: 100
                        contentItem: Text {
                            text: "Тренировать модель"
                            wrapMode: Text.Wrap
                            verticalAlignment: Text.AlignVCenter
                            horizontalAlignment: Text.AlignHCenter
                        }
                        //text: "Тренировать модель"
                        //enabled: !predicter.trainingProcess && predicter.fakePath != "" && predicter.authenticPath != ""
                        //         && predicter.fakePath != predicter.authenticPath
                        //anchors.left: layoutAuthentic.left
                        //anchors.top: layoutAuthentic.bottom
                        //anchors.leftMargin: szNormalMargin
                        //anchors.topMargin: szNormalMargin / 2
                        onClicked: {
                            //predicter.runTraining()
                            trainingPopup.open()
                        }
                    }
                    Button {
                        Layout.preferredHeight: 50
                        Layout.preferredWidth: 100
                        //text: "Загрузить модель"
                        enabled: !predicter.trainingProcess
                        contentItem: Text {
                            text: "Загрузить модель"
                            wrapMode: Text.Wrap
                            verticalAlignment: Text.AlignVCenter
                            horizontalAlignment: Text.AlignHCenter
                        }
                        //anchors.left: btnTrainModel.right
                        //anchors.top: layoutAuthentic.bottom
                        //anchors.leftMargin: szNormalMargin
                        //anchors.topMargin: szNormalMargin / 2
                        onClicked: {
                            loadModelDialog.open()
                        }
                    }

                    Button {
                        //implicitWidth: 50
                        Layout.preferredHeight: 50
                        Layout.maximumWidth: 150
                        //text: qsTr("Выберите изображение для проверки")
                        contentItem: Text {
                            text: "Выберите изображение для проверки"
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

//            Button {
//                //implicitWidth: 50
//                Layout.preferredHeight: 50
//                text: qsTr("Hello")
//                enabled: !predicter.trainingProcess && predicter.modelPrepared
//                onClicked: {
//                    console.log("B")
//                    imagePredictDialog.open()
//                    //predicter.predictImage("C:\\Users\\imynn\\Downloads\\CASIA2\\Au\\Au_ani_101899.jpg")
//                }
//            }
//        }

//        ColumnLayout {
//            id: layoutAuthentic
//            anchors.left: verificationLayout.right
//            anchors.right: parent.right

//            Text {
//                text: "Укажите путь до подлинных изображений"
//            }

//            RowLayout {
//                Layout.fillWidth: true

//                TextField {
//                    Layout.fillWidth: true
//                    readOnly: true

//                    Text {
//                        anchors.fill: parent
//                        text: predicter.authenticPath
//                        elide: Text.ElideRight
//                    }
//                }

//                Button {
//                    text: "Выбрать папку"
//                    enabled: !predicter.trainingProcess
//                    onClicked: {
//                        authenticFolderDialog.open()
//                    }
//                }
//            }

//            Text {
//                text: "Укажите путь до поддельных изображений"
//            }

//            RowLayout {
//                Layout.fillWidth: true

//                TextField {
//                    Layout.fillWidth: true
//                    readOnly: true

//                    Text {
//                        anchors.fill: parent
//                        text: predicter.fakePath
//                        elide: Text.ElideRight
//                    }
//                }

//                Button {
//                    text: "Выбрать папку"
//                    enabled: !predicter.trainingProcess
//                    onClicked: {
//                        fakeFolderDialog.open()
//                    }
//                }
//            }
//        }


//        Button {
//            id: btnTrainModel
//            height: 50
//            text: "Тренировать модель"
//            //enabled: !predicter.trainingProcess && predicter.fakePath != "" && predicter.authenticPath != ""
//            //         && predicter.fakePath != predicter.authenticPath
//            anchors.left: layoutAuthentic.left
//            anchors.top: layoutAuthentic.bottom
//            //anchors.leftMargin: szNormalMargin
//            anchors.topMargin: szNormalMargin / 2
//            onClicked: {
//                //predicter.runTraining()
//                trainingPopup.open()
//            }
//        }
//        Button {
//            height: 50
//            text: "Загрузить модель"
//            enabled: !predicter.trainingProcess
//            anchors.left: btnTrainModel.right
//            anchors.top: layoutAuthentic.bottom
//            anchors.leftMargin: szNormalMargin
//            anchors.topMargin: szNormalMargin / 2
//            onClicked: {
//                loadModelDialog.open()
//            }
//        }

        Popup {
            id: trainingPopup
            anchors.centerIn: parent
            //implicitHeight: parent.height // (parent.height / 2) - szNormalMargin
            //implicitWidth: parent.width //(parent.width / 2) - szNormalMargin
            //implicitHeight: zz.implicitHeight + 2 * szNormalMargin
            implicitWidth: 300
            implicitHeight: 200 + szNormalMargin
            ColumnLayout {
                id: zz
                //anchors.fill: parent
                //anchors.left: verificationLayout.right
                //anchors.left: parent.left
                //anchors.right: parent.right
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

                        }

                        Text {
                            text: "Число батчей"
                        }

                        TextField {
                            Text {

                            }
                        }
                    }

                    Button {
                        Layout.topMargin: szNormalMargin
                        Layout.alignment: Qt.AlignVCenter
                        Layout.preferredHeight: 50
                        text: "Начать обучение модели"
                        onClicked: {
                            trainingPopup.close()
                        }
                    }
                }
            }
        }

        FileDialog {
            id: loadModelDialog
            onAccepted: {
                predicter.loadModel(selectedFile.toString())
               // predictImage(selectedFile)
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
            onAccepted: {
                predicter.imagePath = selectedFile.toString()
                predicter.predictImage(selectedFile)
               // predictImage(selectedFile)
            }
        }
    }
}
