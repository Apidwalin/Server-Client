"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const vscode_1 = require("vscode");
function activate(context) {
    const workspaceLabel = new WorkspaceLabel();
    context.subscriptions.push(workspaceLabel);
}
exports.activate = activate;
// this method is called when your extension is deactivated
function deactivate() { }
exports.deactivate = deactivate;
class WorkspaceLabel {
    constructor() {
        this.statusBarItem = vscode_1.window.createStatusBarItem(vscode_1.StatusBarAlignment.Right);
        this.workspaceFolderName = vscode_1.workspace.workspaceFolders;
        this.updateWorkSpaceLabel();
    }
    updateWorkSpaceLabel() {
        if (this.workspaceFolderName) {
            this.statusBarItem.color = '#D73900';
            this.statusBarItem.text = `[ ${this.workspaceFolderName[0].name} ]`;
            this.statusBarItem.show();
        }
    }
    dispose() {
        this.statusBarItem.dispose();
    }
}
exports.WorkspaceLabel = WorkspaceLabel;
//# sourceMappingURL=extension.js.map