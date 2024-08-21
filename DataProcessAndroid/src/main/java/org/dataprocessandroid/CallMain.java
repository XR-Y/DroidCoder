package org.callgraph;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.File;
import java.util.*;

public class CallMain {
    private static final Logger logger = LogManager.getLogger(CallMain.class);

    public static void main(String[] args) {
        logger.debug("Start CallMain");
        String[] result = Utils.getSimilarBaseDirs();
        String flag = "test";
        String fpath = "call_tasks_" + flag + ".json";
        Map<String, String[]> callTasks = Utils.getCallTasks(fpath);
        Map<String, List<String[]>> tasks = Utils.getSimTasks("java", flag, callTasks.keySet());
        callTasks.putAll(new JavaCallFinder(result, tasks).executeCallFinding());
        // Utils.writeCallJsonFile(callTasks, "call_tasks_java.json");
        tasks = Utils.getSimTasks("kt", flag, callTasks.keySet());
        // Map<String, String[]> callTasksKt = new HashMap<>();
        for (String res: result) {
            callTasks.putAll(new KotlinCallFinder().executeCallFinding(new String[]{res}, tasks, fpath));
            Utils.writeCallJsonFile(callTasks, fpath);
        }
        // logger.info("Java Call Tasks Count: " + callTasks.size());
        // callTasks.putAll(callTasksKt);
        logger.info("Total Call Tasks Count: " + callTasks.size());
        Utils.writeCallJsonFile(callTasks, fpath);
    }

    private static String[] listDirectories(String[] dirs) {
        return Arrays.stream(dirs)
                .flatMap(dir -> Arrays.stream(Objects.requireNonNull(new File(dir).listFiles(File::isDirectory))))
                .map(File::getAbsolutePath)
                .toArray(String[]::new);
    }

}