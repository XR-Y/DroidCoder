package org.callgraph;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.File;
import java.nio.file.Paths;
import java.util.*;

import static org.callgraph.Utils.getAllImports;
import static org.callgraph.Utils.getAndroidImports;

public class Classifier {
    private static final Logger logger = LogManager.getLogger(Classifier.class);
    private static final String[] flags = new String[]{"new", "test"};
    private static final String baseDir = Paths.get(".")
                                         .getParent()
                                         .resolve("datasets")
                                         .resolve(flags[0])
                                         .resolve(flags[1])
                                         .toString();
    private static final String fileName = "dataset_" + flags[0] + "_" + flags[1] + ".json";

    public static void main (String[] args) {
        new AndroidManifestParser(baseDir).collectPermissions();
        // new AndroidManifestParser(baseDir).dumpAllActions();
        List<Map<String, List<String>>> result = new AndroidManifestParser(baseDir).execute();
        Map<String, List<String>> androidMethodsMap = result.get(0); // repo -> four types of components
        Map<String, List<String>> methodActionMap = result.get(1); // component -> actions
        List<Map<String, String>> tasks = Utils.getOriginTasks(fileName);
        Map<String, List<String>> methodImports = new HashMap<>();
        int idx = 0;
        for (Map<String, String> task : tasks) {
            String filePath = task.get("fpath");
            String repo = filePath.substring(0, filePath.indexOf(File.separator));
            List<String> documents = androidMethodsMap.get(repo);
            if (documents == null) {
                task.put("type", "java");
                task.put("actions", "");
                continue;
            }
            String methodName = task.get("methodName");
            if (methodName == null || methodName.isEmpty()) {
                methodName = Utils.extractMethodName(task.get("prompt") + task.get("ground_truth"),
                        filePath.substring(filePath.lastIndexOf(".") + 1));
                task.put("methodName", methodName);
            }
            String curPath = baseDir + File.separator + filePath;
            for (String document : documents) {
                if (checkNameImports(curPath, document, methodImports, task.get("prompt"), methodName)) {
                    task.put("type", "android");
                    idx += 1;
                    break;
                }
            }
            task.putIfAbsent("type", "java");
            for (String document : methodActionMap.keySet()) {
                if (filePath.contains(document)) {
                    List<String> actions = methodActionMap.get(document);
                    String actionString = String.join("\n", actions);
                    task.put("actions", actionString);
                }
            }
            task.putIfAbsent("actions", "");
        }
        methodImports.clear();
        System.out.println("Android Methods Count: " + idx + "/" + tasks.size());
        for (int i = 0; i < tasks.size(); i++) {
            Map<String, String> task = tasks.get(i);
            String filePath = task.get("fpath");
            List<String> imports = getAllImports(baseDir + File.separator + filePath, methodImports);
            StringBuilder prompt = new StringBuilder(task.get("prompt"));
            for (String imp: imports) {
                if (prompt.toString().contains(imp.lastIndexOf(".") == -1 ? imp : imp.substring(imp.lastIndexOf(".") + 1))) {
                    prompt.insert(0, "import " + imp + "\n");
                }
            }
            task.put("prompt", prompt.toString());
            if ((i + 1) % 50 == 0) System.out.println("Processing: " + (i + 1) + "/" + tasks.size());
        }
        Utils.saveDatasets(tasks, fileName);
    }

    private static boolean checkNameImports (String filePath, String document, Map<String, List<String>> methodImports, String code, String methodName) {
        if (methodName.toLowerCase().startsWith("on") || code.contains("@Composable")) {
            return true;
        }
        else if (!filePath.contains(document)) {
            return false;
        }
        List<String> imports = getAndroidImports(filePath, methodImports);
        if (imports.isEmpty()) {
            return true;
        }
        for (String imp : imports) {
            if (code.contains(imp)) {
                return true;
            }
        }
        return false;
    }

}
