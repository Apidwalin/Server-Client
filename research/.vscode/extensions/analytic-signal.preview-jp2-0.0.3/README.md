# JPEG-2000 Preview in VSCode

Simple preview for JPEG 2000 (`.jp2`, `.j2k`) files in VSCode. The extension replicates the same image pan and zoom behaviour as VSCode's built-in Image Editor. Like the Image Editor, the status bar is used to report the size of the file on disk, the image dimensions in pixels, and the current zoom level.

![JPEG-2000 Preview](https://www.analyticsignal.com/images/vscode-jp2-preview.png)  
*JPEG-2000 Preview of high resolution (33 megapixel) photograph of drill cores. Contains British Geological Survey materials &copy; UKRI 2021.*

## Usage

### **Click**

To preview a file:

- Click on a (`.jp2`, `.j2k`) file 

A preview of the file contents should now be shown in an editor.

### **Open to the Side**

To preview a file in a new editor (if other editors are already open):

- Right-click on a file with (`.jp2` or `.j2k`) extension and choose *Open to the Side*

A preview of the file contents should now be shown in a new editor.

### **Open With...**

To choose how to view a TIFF file:

- Right-click on a file with (`.jp2` or `.j2k`) extension and choose *Open With...*
- A list of available options appears in the Command Palette, choose one of the options which will include:

    > *JPEG-2000 Preview*, the viewer provided by this extension. Choose this option to preview the file.

### **Zoom** 

You can control the zoom level of the image:

- Click in the editor to zoom in, or shift-click (Windows)  option-click (macOS) to zoom out. The current zoom level is shown in the status bar.
  
- Click on the zoom level in the status bar to reveal a selection list in the Command Palette. Choose the desired zoom level from the list.