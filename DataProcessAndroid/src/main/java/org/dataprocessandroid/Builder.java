package org.callgraph;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import static org.callgraph.Utils.getAllImports;

public class Builder {
    private final static Logger logger = LogManager.getLogger(Builder.class);
    private final static String baseDir = ""; // path to the root directory of the repos
    private final static Random random = new Random(42);

    public static void main(String[] args) {
        // dumpAllMethods();
        Map<String, Map<String, List<String>>> repos = Utils.getRepos(baseDir);
        String[] flags = new String[]{"train", "test"};
        boolean lineLevel = false;
        for (String appType : repos.keySet()) {
            // if (!appType.equals("new")) continue;
            for (String flag : flags) {
                List<String> dirs = repos.get(appType).get(flag);
                dirs = dirs.subList(7, dirs.size());
                List<Map<String, String>> results = new ArrayList<>();
                results = Utils.getOriginTasks("dataset_" + appType + "_" + flag + ".json");
                for (String dir : dirs) {
                    results.addAll(iterateRepo(dir, flag, appType, lineLevel));
                }
                logger.info("Got " + appType + " " + flag + " " + results.size());
                String fileName = "dataset_" + appType + "_" + flag + ".json";
                if (lineLevel) fileName = "line_" + fileName;
                Utils.saveDatasets(results, fileName);
            }
        }
    }

    private static List<Map<String, String>> iterateRepo(String dir, String flag, String appType, boolean lineLevel) {
        List<Map<String, String>> results = new ArrayList<>();
        try (Stream<Path> walk = Files.walk(Paths.get(baseDir, appType, flag, dir))) {
            List<String> result = walk.filter(Files::isRegularFile).map(Path::toString)
                    .filter(string -> string.endsWith(".java") || string.endsWith(".kt"))
                    .collect(Collectors.toList());
            for (int i = 0; i < result.size(); i++) {
                String file = result.get(i);
                List<String[]> methods;
                if (file.endsWith(".java")) methods = new JavaCallFinder().extractAllMethods(file);
                else methods = new KotlinCallFinder().extractAllMethods(file);
                if (methods.isEmpty()) continue;
                String[] variables = methods.remove(methods.size() - 1); // last element is global variables
                for (String[] method : methods) {
                    Map<String, String> map = new HashMap<>();
                    int cnt = -5;
                    if (method[1].split("\n").length < 6) cnt = -method[1].split("\n").length;
                    if (lineLevel) {
                        // method[0] is the context with imports, method[1] is the method body
                        String imports = method[0].substring(0, method[0].indexOf(method[1]));
                        String[] lines = method[1].split("\n");
                        cnt = Builder.random.nextInt(Math.max(lines.length - 3, 1)) + 1;
                        map.put("prompt", imports + String.join("\n", Arrays.copyOfRange(lines, 0, cnt)));
                        map.put("ground_truth", String.join("\n", new String[]{lines[cnt]}));
                        assert map.get("ground_truth").length() * map.get("prompt").length() > 0; // check if prompt and ground_truth are not empty
                    } else {
                        String[] lines = method[0].split("\n");
                        map.put("prompt", String.join("\n", Arrays.copyOfRange(lines, 0, lines.length + cnt)));
                        map.put("ground_truth", String.join("\n", Arrays.copyOfRange(lines, lines.length + cnt, lines.length)));
                    }
                    map.put("fpath", file.substring(file.indexOf(Paths.get(dir).toString())));
                    map.put("context", method[0]);
                    map.put("task_id", dir.substring(0, dir.indexOf("/")) + "/" + results.size());
                    map.put("start_lineno", method[2]);
                    map.put("end_lineno", method[3]);
                    map.put("methodName", method[4]);
                    map.put("app_type", appType);
                    map.put("variables", String.join("\n", variables));
                    results.add(map);
                }
                if ((i + 1) % 20 == 0) logger.info("Processed " + dir + " " + (i + 1) + "/" + result.size());
             }
        } catch (IOException e) {
            logger.error("exception message", e);
        }
        return results;
    }

    private static void dumpAllMethods() {
        Map<String, Map<String, List<String>>> repos = Utils.getRepos(baseDir);
        Map<String, List<Map<String, String>>> outputs = new HashMap<>();
        outputs = Utils.loadAllMethods();
        Map<String, List<String>> methodImports = new HashMap<>();
        for (String appType : repos.keySet()) {
            // if (!appType.equals("communication")) continue;
            List<String> dirs = repos.get(appType).get("train");
            List<Map<String, String>> results = new ArrayList<>();
            for (String dir : dirs) {
                try (Stream<Path> walk = Files.walk(Paths.get(baseDir, appType, "train", dir))) {
                    List<String> result = walk.filter(Files::isRegularFile).map(Path::toString)
                            .filter(string -> string.endsWith(".java") || string.endsWith(".kt"))
                            .collect(Collectors.toList());
                    for (int i = 0; i < result.size(); i++) {
                        String file = result.get(i);
                        List<String[]> methods;
                        if (file.endsWith(".java")) methods = new JavaCallFinder().extractAllMethods(file);
                        else methods = new KotlinCallFinder().extractAllMethods(file);
                        if (methods.isEmpty()) continue;
                        String[] variables = methods.remove(methods.size() - 1); // last element is global variables
                        List<String> imports = getAllImports(file, methodImports);
                        for (String[] method : methods) {
                            Map<String, String> map = new HashMap<>();
                            map.put("fpath", file.substring(file.indexOf(Paths.get(dir).toString())));
                            map.put("methodName", method[4]);
                            map.put("repo", dir.substring(0, dir.indexOf("/")));
                            StringBuilder prompt = new StringBuilder(method[0]);
                            for (String imp: imports) {
                                if (prompt.toString().contains(imp.lastIndexOf(".") == -1 ? imp : imp.substring(imp.lastIndexOf(".") + 1))) {
                                    prompt.insert(0, "import " + imp + "\n");
                                }
                            }
                            map.put("context", prompt.toString());
                            results.add(map);
                        }
                        if ((i + 1) % 20 == 0) System.out.println("Processed " + dir + " " + (i + 1) + "/" + result.size());
                     }
                } catch (IOException e) {
                    logger.error("exception message", e);
                }
            }
            outputs.put(appType, results);
            Utils.saveAllMethods(outputs, "all_methods.json");
        }
        Utils.saveAllMethods(outputs, "all_methods.json");
    }
}
