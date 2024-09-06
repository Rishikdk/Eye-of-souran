from PyQt5 import QtWidgets, QtGui, QtCore
import pandas as pd
import os

def show_notifications(parent_widget):
    """Read the CSV file and display the information in a table format."""
    log_path = 'detection_log.csv'
    
    if not os.path.exists(log_path):
        QtWidgets.QMessageBox.warning(parent_widget, "No Data", "No detection data found.")
        return

    # Read CSV file, assuming it doesn't have headers and includes an image path
    df = pd.read_csv(log_path, header=None, names=['Camera Index', 'Name', 'Timestamp', 'Image Path'])
    
    # Create a new widget to display the table
    table_widget = QtWidgets.QWidget()
    table_layout = QtWidgets.QVBoxLayout(table_widget)

    # Create QTableWidget
    table = QtWidgets.QTableWidget()
    table.setRowCount(len(df))
    table.setColumnCount(4)  # We have 4 columns: Camera Index, Name, Timestamp, Image
    
    # Set column headers
    column_headers = ['Camera Index', 'Name', 'Timestamp', 'Image']
    table.setHorizontalHeaderLabels(column_headers)

    # Set column widths
    table.setColumnWidth(0, 100)  # Camera Index
    table.setColumnWidth(1, 150)  # Name
    table.setColumnWidth(2, 200)  # Timestamp
    table.setColumnWidth(3, 150)  # Image

    # Set row height to match image size (3x3 cm in pixels, assuming 96 dpi)
    image_size = QtCore.QSize(90, 90)  # 3x3 cm ≈ 90x90 pixels at 96 dpi

    # Apply a custom stylesheet for better appearance
    table.setStyleSheet("""
        QTableWidget::item {
            padding: 10px;
            border: none;
            font-size: 14px;
            color: #333;
        }
        QHeaderView::section {
            color: black;
            padding: 5px;
            font-size: 16px;
            border: 1px solid #ddd;
        }
        QTableWidget {
            gridline-color: #ddd;
            font-family: Arial, Helvetica, sans-serif;
            border: 1px solid #ddd;
        }
    """)

    # Populate table with data
    for row_idx, row in df.iterrows():
        table.setItem(row_idx, 0, QtWidgets.QTableWidgetItem(str(row['Camera Index'])))
        table.setItem(row_idx, 1, QtWidgets.QTableWidgetItem(row['Name']))
        table.setItem(row_idx, 2, QtWidgets.QTableWidgetItem(row['Timestamp']))

        # Load and display the image
        image_label = QtWidgets.QLabel()
        image = QtGui.QPixmap(row['Image Path']).scaled(image_size, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        image_label.setPixmap(image)
        table.setCellWidget(row_idx, 3, image_label)
        table.setRowHeight(row_idx, image_size.height() + 20)  # Add padding for better appearance

    table_layout.addWidget(table)

    # Create a new window for the table
    table_window = QtWidgets.QWidget()
    table_window.setWindowTitle("Detection Notifications")
    table_window.setLayout(table_layout)
    table_window.resize(800, 600)  # Adjust size as needed
    table_window.show()

    # Keep the reference to the window so it doesn't get garbage-collected
    parent_widget.table_window = table_window
