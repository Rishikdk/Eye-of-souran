from PyQt5 import QtWidgets
from ui_sidebar import Ui_MainWindow
from dashboard import Ui_Dashboard
from search import Ui_SearchApp
from notifications import show_notifications  
from video import VideoSearchApp

class SidebarApp(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setupUi(self)
        print("UI setup complete")

        self.dashboard_btn1.clicked.connect(self.show_dashboard)
        self.search_btn1.clicked.connect(self.show_search)
        self.notification_1.clicked.connect(self.show_notifications)  
        self.settings_1.clicked.connect(self.open_video_search_app)


        if self.widget.layout() is None:
            self.widget.setLayout(QtWidgets.QVBoxLayout())

        self.show_dashboard()
    def open_video_search_app(self):
        """Open the VideoSearchApp when the settings_1 button is clicked."""
        self.video_search_app = VideoSearchApp()
        self.video_search_app.show()

    def show_dashboard(self):
        print("Dashboard button clicked")
        self.clear_current_widget()
        self.dashboard_widget = QtWidgets.QWidget()
        self.dashboard_ui = Ui_Dashboard()
        self.dashboard_ui.setupUi(self.dashboard_widget)
        self.current_widget = self.dashboard_widget
        self.widget.layout().addWidget(self.current_widget)

    def show_search(self):
        print("Search button clicked")
        self.clear_current_widget()
        self.search_widget = QtWidgets.QWidget()
        self.search_ui = Ui_SearchApp()
        self.search_ui.setupUi(self.search_widget)
        self.current_widget = self.search_widget
        self.widget.layout().addWidget(self.current_widget)

    def show_notifications(self):
        """Call the function from notifications.py to display the notifications."""
        show_notifications(self)

    
    def clear_current_widget(self):
        """Remove the current widget from the layout."""
        if hasattr(self, 'current_widget') and self.current_widget is not None:
            self.widget.layout().removeWidget(self.current_widget)
            self.current_widget.setParent(None)

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = SidebarApp()
    window.show()
    sys.exit(app.exec_())
