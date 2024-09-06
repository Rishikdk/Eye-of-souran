import sys
from PyQt5.QtWidgets import QApplication
from sidebar import SidebarApp  # Import the SidebarApp class

if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWin = SidebarApp()  # Create an instance of SidebarApp
    mainWin.show()
    sys.exit(app.exec_())
