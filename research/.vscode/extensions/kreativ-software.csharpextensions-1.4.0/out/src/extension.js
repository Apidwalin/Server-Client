"use strict";
var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.deactivate = exports.activate = void 0;
const vscode = require("vscode");
const path = require("path");
const os_1 = require("os");
const fs_1 = require("fs");
const codeActionProvider_1 = require("./codeActionProvider");
const namespaceDetector_1 = require("./namespaceDetector");
function activate(context) {
    const documentSelector = {
        language: 'csharp',
        scheme: 'file'
    };
    context.subscriptions.push(vscode.commands.registerCommand('csharpextensions.createClass', createClass));
    context.subscriptions.push(vscode.commands.registerCommand('csharpextensions.createInterface', createInterface));
    context.subscriptions.push(vscode.commands.registerCommand('csharpextensions.createEnum', createEnum));
    const codeActionProvider = new codeActionProvider_1.default();
    const disposable = vscode.languages.registerCodeActionsProvider(documentSelector, codeActionProvider);
    context.subscriptions.push(disposable);
}
exports.activate = activate;
function createClass(args) {
    return __awaiter(this, void 0, void 0, function* () {
        yield promptAndSave(args, 'class');
    });
}
function createInterface(args) {
    return __awaiter(this, void 0, void 0, function* () {
        yield promptAndSave(args, 'interface');
    });
}
function createEnum(args) {
    return __awaiter(this, void 0, void 0, function* () {
        yield promptAndSave(args, 'enum');
    });
}
function promptAndSave(args, templatetype) {
    return __awaiter(this, void 0, void 0, function* () {
        if (args == null) {
            args = { _fsPath: vscode.workspace.rootPath };
        }
        const incomingpath = args._fsPath || args.fsPath || args.path;
        try {
            const newfilename = yield vscode.window.showInputBox({ ignoreFocusOut: true, prompt: 'Please enter filename', value: 'new' + templatetype + '.cs' });
            if (typeof newfilename === 'undefined' || newfilename === '')
                return;
            let newfilepath = incomingpath + path.sep + newfilename;
            newfilepath = correctExtension(newfilepath);
            try {
                yield fs_1.promises.access(newfilepath);
                vscode.window.showErrorMessage(`File already exists: ${os_1.EOL}${newfilepath}`);
                return;
            }
            catch (_a) { }
            const namespaceDetector = new namespaceDetector_1.default(newfilepath);
            const namespace = yield namespaceDetector.getNamespace();
            const typename = path.basename(newfilepath, '.cs');
            yield openTemplateAndSaveNewFile(templatetype, namespace, typename, newfilepath);
        }
        catch (errOnInput) {
            console.error('Error on input', errOnInput);
            vscode.window.showErrorMessage('Error on input. See extensions log for more info');
        }
        ;
    });
}
function correctExtension(filename) {
    if (path.extname(filename) !== '.cs') {
        if (filename.endsWith('.'))
            filename = filename + 'cs';
        else
            filename = filename + '.cs';
    }
    return filename;
}
function openTemplateAndSaveNewFile(type, namespace, filename, originalfilepath) {
    return __awaiter(this, void 0, void 0, function* () {
        const templatefileName = type + '.tmpl';
        const extension = vscode.extensions.getExtension('kreativ-software.csharpextensions');
        if (!extension) {
            vscode.window.showErrorMessage('Weird, but the extension you are currently using could not be found');
            return;
        }
        const templateFilePath = path.join(extension.extensionPath, 'templates', templatefileName);
        try {
            const doc = yield fs_1.promises.readFile(templateFilePath, 'utf-8');
            let text = doc
                .replace('${namespace}', namespace)
                .replace('${classname}', filename);
            const cursorPosition = findCursorInTemplate(text);
            text = text.replace('${cursor}', '');
            yield fs_1.promises.writeFile(originalfilepath, text);
            const openedDoc = yield vscode.workspace.openTextDocument(originalfilepath);
            const editor = yield vscode.window.showTextDocument(openedDoc);
            if (cursorPosition != null) {
                const newselection = new vscode.Selection(cursorPosition, cursorPosition);
                editor.selection = newselection;
            }
        }
        catch (errTryingToCreate) {
            const errorMessage = `Error trying to create file '${originalfilepath}' from template '${templatefileName}'`;
            console.error(errorMessage, errTryingToCreate);
            vscode.window.showErrorMessage(errorMessage);
        }
    });
}
function findCursorInTemplate(text) {
    const cursorPos = text.indexOf('${cursor}');
    const preCursor = text.substr(0, cursorPos);
    const matchesForPreCursor = preCursor.match(/\n/gi);
    if (matchesForPreCursor === null)
        return null;
    const lineNum = matchesForPreCursor.length;
    const charNum = preCursor.substr(preCursor.lastIndexOf('\n')).length;
    return new vscode.Position(lineNum, charNum);
}
function deactivate() { }
exports.deactivate = deactivate;
//# sourceMappingURL=extension.js.map