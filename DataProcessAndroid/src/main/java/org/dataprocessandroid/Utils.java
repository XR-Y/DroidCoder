package org.callgraph;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.ObjectReader;
import com.fasterxml.jackson.databind.SerializationFeature;
import org.apache.commons.io.FileUtils;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class Utils {
    private static final Logger logger = LogManager.getLogger(Utils.class);
    private static final String baseDir = "D:\\code\\Sync\\project\\RepoCoder\\AndroidCompletion";
    private static final String predictionsDir = "D:\\code\\Sync\\project\\RepoCoder\\predictions";
    private static final String datasetsDir = Paths.get(".").getParent().resolve("datasets").toString();
    private static final String datasetPath = baseDir + File.separator + "datasets";
    private static final String callTasksPath = baseDir + File.separator + "resources";

    public static List<Map<String, String>> getOriginTasks (String fileName) {
        String tasksPath = datasetPath + File.separator + fileName;
        return getTasks(tasksPath);
    }

    public static void saveDatasets (List<Map<String, String>> tasks, String fileName) {
        String tasksPath = datasetPath + File.separator + fileName;
        saveJsonFile(tasks, tasksPath);
    }

    public static void saveActionsMap (Map<String, List<String>> actionsMap, String fileName) {
        String tasksPath = datasetPath + File.separator + fileName;
        saveJsonFile(actionsMap, tasksPath);
    }

    public static void saveAllMethods (Map<String, List<Map<String, String>>> tasks, String fileName) {
        String tasksPath = datasetPath + File.separator + fileName;
        saveJsonFile(tasks, tasksPath);
    }

    public static List<Map<String, String>> getOutputTasks (String path) {
        return getTasks(path);
    }

    public static Map<String, List<Map<String, String>>> getAllOutputTasks () {
        try (Stream<Path> walk = Files.walk(Paths.get(predictionsDir))) {
            List<String> result = walk.filter(Files::isRegularFile)
                    .filter(path -> !path.getParent().getFileName().toString().equals("past"))
                    .map(Path::toString)
                    .filter(filename -> filename.endsWith(".jsonl") && !filename.contains("unknown"))
                    .collect(Collectors.toList());
            Map<String, List<Map<String, String>>> results = new HashMap<>();
            for (String file : result) {
                String key = file.substring(file.lastIndexOf(File.separator) + 1);
                results.put(key, getOutputTasks(file));
            }
            return results;
        } catch (IOException e) {
            logger.error("exception message", e);
        }
        return Collections.emptyMap();
    }

    private static List<Map<String, String>> getTasks (String tasksPath) {
        List<Map<String, String>> tasks = new ArrayList<>();
        if (tasksPath.endsWith(".jsonl")) return readJsonLines(tasksPath);
        JsonNode jsonNode = readJson(tasksPath);
        assert jsonNode != null && jsonNode.isArray();
        ObjectMapper objectMapper = new ObjectMapper();
        try {
            tasks = objectMapper.readValue(jsonNode.traverse(), new TypeReference<List<Map<String, String>>>(){});
        } catch (Exception e) {
            logger.error("exception message", e);
        }
        return tasks;
    }

    public static Map<String, String[]> getCallTasks (String fileName) {
        Map<String, String[]> tasks = new HashMap<>();
        String tasksPath = callTasksPath + File.separator + fileName;
        if (!new File(tasksPath).exists()) {
            logger.info("File not found: " + tasksPath + ", execute all tasks.");
            return tasks;
        }
        JsonNode jsonNode = readJson(tasksPath);
        assert jsonNode != null && jsonNode.isArray();
        ObjectMapper objectMapper = new ObjectMapper();
        try {
            tasks = objectMapper.readValue(jsonNode.traverse(), new TypeReference<Map<String, String[]>>(){});
        } catch (Exception e) {
            logger.error("exception message", e);
        }
        return tasks;
    }

    public static Map<String, List<String[]>> getSimTasks (String type, String flag, Set<String> ids) {
        String tasksPath = datasetPath + File.separator + "dataset_sim_" + flag + ".json";
        JsonNode jsonNode = readJson(tasksPath);
        assert jsonNode != null && jsonNode.isArray();
        Map<String, List<String[]>> taskMap = new HashMap<>();
        for (JsonNode task : jsonNode) {
            if (ids.contains(task.get("task_id").asText())) continue;
            if (task.get("similar_function_file") == null) {
                continue;
            }
            if(type.equals("kt") && !task.get("similar_function_file").asText().endsWith(".kt")) continue;
            if(type.equals("java") && !task.get("similar_function_file").asText().endsWith(".java")) continue;
            String repo = task.get("similar_function_repo").asText();
            String filePath = task.get("similar_function_file").asText();
            filePath = filePath.substring(filePath.indexOf(File.separator) + 1);
            String[] method = new String[]{task.get("similar_function_context").asText(), filePath,
                    task.get("task_id").asText(), task.get("similar_function_name").asText(), task.get("type").asText()};
            if (taskMap.containsKey(repo)) {
                taskMap.get(repo).add(method);
            } else {
                List<String[]> list = new ArrayList<>();
                list.add(method);
                taskMap.put(repo, list);
            }
        }
        return taskMap;
    }

    public static String readFile (String filePath) {
        try {
            FileInputStream fileInputStream = new FileInputStream(filePath);
            byte[] buffer = new byte[fileInputStream.available()];
            fileInputStream.read(buffer);
            fileInputStream.close();
            return new String(buffer);
        } catch (IOException e) {
            logger.error("exception message", e);
        }
        return "";
    }

    private static JsonNode readJson (String filePath) {
        JsonNode jsonNode = null;
        try {
            ObjectMapper objectMapper = new ObjectMapper();
            jsonNode = objectMapper.readTree(Paths.get(filePath).toFile());
        } catch (IOException e) {
            logger.error("exception message", e);
        }
        return jsonNode;
    }

    private static List<Map<String, String>> readJsonLines(String filePath) {
        ObjectMapper objectMapper = new ObjectMapper();
        List<Map<String, String>> dataList = new ArrayList<>();
        try (BufferedReader reader = new BufferedReader(new FileReader(filePath))) {
            String line;
            while ((line = reader.readLine()) != null) {
                JsonNode jsonNode = objectMapper.readTree(line);
                Map<String, String> data = new HashMap<>();
                jsonNode.fields().forEachRemaining(entry -> {
                    String key = entry.getKey();
                    JsonNode valueNode = entry.getValue();
                    if (entry.getValue().isArray() && entry.getValue().get(0).has("text")) {
                        data.put(entry.getKey(), entry.getValue().get(0).get("text").asText());
                    } else if (entry.getKey().equals("metadata") && entry.getValue().has("task_id")) {
                        data.put("task_id", entry.getValue().get("task_id").asText());
                        data.put("fpath", entry.getValue().get("fpath").asText());
                    } else {
                        data.put(entry.getKey(), entry.getValue().asText());
                    }
                });
                dataList.add(data);
            }
        } catch (IOException e) {
            logger.error("exception message", e);
        }
        return dataList;
    }

    private static String convertDataToJsonString(Object data) {
        try {
            ObjectMapper objectMapper = new ObjectMapper();
            objectMapper.enable(SerializationFeature.INDENT_OUTPUT);
            logger.debug("start converting data to JSON string");
            return objectMapper.writeValueAsString(data);
        } catch (IOException e) {
            logger.error("exception message", e);
            return null;
        }
    }

    private static void saveJsonStringToFile(String jsonString, String filePath) {
        try {
            File file = new File(filePath);
            FileUtils.writeStringToFile(file, jsonString, "UTF-8");
            logger.info("JSON saved to: " + file.getAbsolutePath());
        } catch (IOException e) {
            logger.error("exception message", e);
        }
    }

    public static void saveJsonFile(Object data, String filePath) {
        String jsonString = convertDataToJsonString(data);
        saveJsonStringToFile(jsonString, filePath);
    }

    public static void writeCallJsonFile(Object data, String filePath) {
        try (OutputStream out = Files.newOutputStream(Paths.get(callTasksPath + File.separator + filePath))) {
            ObjectMapper objectMapper = new ObjectMapper();
            objectMapper.enable(SerializationFeature.INDENT_OUTPUT);
            logger.debug("start converting data to JSON string");
            objectMapper.writer().writeValue(out, data);
        } catch (IOException e) {
            logger.error("exception message", e);
        }
    }

    public static String extractMethodName (String code, String type) {
        String methodName = "";
        try {
            if (Objects.equals(type, "java")) {
                methodName = JavaCallFinder.getMethodName(code);
            } else if (Objects.equals(type, "kt")) {
                methodName = KotlinCallFinder.getMethodName(code);
            }
        } catch (Exception e) {
            logger.error("exception message", e);
        }
        return methodName == null ? "" : methodName;
    }

    public static void updateCallTasks () {
        JsonNode past = readJson("call_tasks_test.json");
        JsonNode cur = readJson("new_call_tasks_test.json");
        ObjectMapper objectMapper = new ObjectMapper();
        Map<String, String[]> callTasks;
        Map<String, String[]> callTasks_past;
        try {
            callTasks = objectMapper.readValue(cur.traverse(), new TypeReference<Map<String, String[]>>(){});
            callTasks_past = objectMapper.readValue(past.traverse(), new TypeReference<Map<String, String[]>>(){});
            for (String key : callTasks_past.keySet()) {
                if (!callTasks.containsKey(key)) {
                    callTasks.put(key, callTasks_past.get(key));
                }
            }
            writeCallJsonFile(callTasks, "call_tasks_test.json");
            logger.info("Total Call Tasks Count: " + callTasks.size());
        } catch (Exception e) {
            logger.error("exception message", e);
        }
    }

    public static Map<String, Map<String, List<String>>> getRepos (String dirs) {
        JsonNode jsonNode = readJson(dirs + File.separator + "repos.json");
        assert jsonNode != null;
        Map<String, Map<String, List<String>>> repos = new HashMap<>();
        try {
            ObjectMapper objectMapper = new ObjectMapper();
            repos = objectMapper.readValue(jsonNode.traverse(), new TypeReference<Map<String, Map<String, List<String>>>>(){});
        } catch (Exception e) {
            logger.error("exception message", e);
        }
        return repos;
    }

    public static String[] getSimilarBaseDirs () {
        Map<String, Map<String, List<String>>> repos = getRepos(datasetsDir);
        List<String> dirs = new ArrayList<>();
        for (String appType : repos.keySet()) {
            for (String dir: repos.get(appType).get("train")) {
                String path = datasetsDir + File.separator + appType + "/train/" + dir;
                dirs.add(Paths.get(path).toString());
            }
        }
        return dirs.toArray(new String[0]);
    }

    public static Map<String, List<Map<String, String>>> loadAllMethods () {
        String tasksPath = datasetPath + File.separator + "all_methods.json";
        JsonNode jsonNode = readJson(tasksPath);
        assert jsonNode != null && jsonNode.isArray();
        Map<String, List<Map<String, String>>> tasks = new HashMap<>();
        try {
            ObjectMapper objectMapper = new ObjectMapper();
            tasks = objectMapper.readValue(jsonNode.traverse(), new TypeReference<Map<String, List<Map<String, String>>>>(){});
        } catch (Exception e) {
            logger.error("exception message", e);
        }
        return tasks;
    }

    private static List<String> getImports(String filePath, Map<String, List<String>> methodImports, Function<String, List<String>> importFinder) {
        if (methodImports.containsKey(filePath)) {
            return methodImports.get(filePath);
        }
        List<String> imports = importFinder.apply(filePath);
        methodImports.put(filePath, imports);
        return imports;
    }

    public static List<String> getAndroidImports(String filePath, Map<String, List<String>> methodImports) {
        return getImports(filePath, methodImports, file -> {
            if (file.endsWith(".java")) {
                return JavaCallFinder.findAndroidImports(file);
            } else if (file.endsWith(".kt")) {
                return KotlinCallFinder.findAndroidImports(file);
            } else {
                logger.error("Unknown file type: " + file);
                return Collections.emptyList();
            }
        });
    }

    public static List<String> getAllImports(String filePath, Map<String, List<String>> methodImports) {
        return getImports(filePath, methodImports, file -> {
            if (file.endsWith(".java")) {
                return JavaCallFinder.getAllImports(file);
            } else if (file.endsWith(".kt")) {
                return KotlinCallFinder.getAllImports(file);
            } else {
                logger.error("Unknown file type: " + file);
                return Collections.emptyList();
            }
        });
    }
}
