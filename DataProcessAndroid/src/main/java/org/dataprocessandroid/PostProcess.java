package org.callgraph;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.*;
import java.util.stream.Collectors;

public class PostProcess {
    private final Logger logger = LogManager.getLogger(PostProcess.class);
    private final List<Map<String, String>> tasks;
    private final List<Map<String, String>> wrongTasks;
    private final JavaCallFinder javaCallFinder;
    private final KotlinCallFinder kotlinCallFinder;
    private final String indexPrompt = "# Determine if the information above is useful, Complete The Following Android Code:";

    public PostProcess (List<Map<String, String>> tasks) {
        this.tasks = tasks;
        this.wrongTasks = new ArrayList<>();
        this.javaCallFinder = new JavaCallFinder();
        this.kotlinCallFinder = new KotlinCallFinder();
    }

    public static void main (String[] args) {
        Map<String, List<Map<String, String>>> outputs = Utils.getAllOutputTasks();
        Map<String, Double> scores = new HashMap<>();
        for (String path : outputs.keySet()) {
            if (outputs.get(path).isEmpty()) continue;
            PostProcess postProcess = new PostProcess(outputs.get(path));
            postProcess.checkBrackets();
            postProcess.checkReturn();
            postProcess.checkVariable();
            scores.put(path, (double) postProcess.wrongTasks.size() / postProcess.tasks.size());
        }
        scores.forEach((k, v) -> System.out.println(k + ": " + (1 - v)));
    }

    private static void dumpOne (String path) {
        String filePath = "wrong_tasks.json";
        PostProcess postProcess = new PostProcess(Utils.getOutputTasks(path));
        postProcess.checkBrackets();
        postProcess.checkReturn();
        postProcess.checkVariable();
        Utils.saveJsonFile(postProcess.wrongTasks, filePath);
    }

    private void checkVariable () {
        List<String> wrongTaskIDs = this.wrongTasks.stream().filter(map -> map.containsKey("task_id"))
                                                            .map(map -> map.get("task_id")).collect(Collectors.toList());
        for (Map<String, String> task : this.tasks) {
            if (wrongTaskIDs.contains(task.get("task_id"))) continue;
            String prompt = task.get("prompt");
            if (prompt.contains(indexPrompt)) {
                prompt = prompt.substring(prompt.indexOf(indexPrompt) + indexPrompt.length());
            }
            String code = formatOutput(prompt + '\n' + task.get("choices"));
            boolean result = true;
            if (task.get("fpath").endsWith(".java")) { // check java unused variables
                result = this.javaCallFinder.checkJavaVariables(code);
                if (!result) result = this.javaCallFinder.checkJavaVariables(formatOutput(task.get("choices")));
            } else if (task.get("fpath").endsWith(".kt")) { // check kotlin unused variables
                result = this.kotlinCallFinder.checkKotlinVariables(code);
                if (!result) result = this.kotlinCallFinder.checkKotlinVariables(formatOutput(task.get("choices")));
            } else {
                logger.error("Unknown file type: " + task.get("fpath"));
            }
            if (!result) {
                String prePrompt = "# You just completed an Android code completion task:\n This is the code you need to complete:\n";
                prompt = prePrompt + prompt + "\n# This is the code you completed:\n" + task.get("choices");
                task.put("prompt", prompt + "\n# However, there is at least one variable, which is only defined but not used. Please check the code and try again.");
                this.wrongTasks.add(task);
            }
        }
    }

    private void checkReturn () {
        List<String> wrongTaskIDs = this.wrongTasks.stream().filter(map -> map.containsKey("task_id"))
                                                            .map(map -> map.get("task_id")).collect(Collectors.toList());
        for (Map<String, String> task : this.tasks) {
            if (wrongTaskIDs.contains(task.get("task_id"))) continue;
            String prompt = task.get("prompt");
            if (prompt.contains(indexPrompt)) {
                prompt = prompt.substring(prompt.indexOf(indexPrompt) + indexPrompt.length());
            }
            String code = formatOutput(prompt + '\n' + task.get("choices"));
            boolean result = true;
            if (task.get("fpath").endsWith(".java")) { // check java return
                result = this.javaCallFinder.checkJavaReturn(code);
                if (!result) result = this.javaCallFinder.checkJavaReturn(formatOutput(task.get("choices")));
            } else if (task.get("fpath").endsWith(".kt")) { // check kotlin return
                result = this.kotlinCallFinder.checkKotlinReturn(code);
                if (!result) result = this.kotlinCallFinder.checkKotlinReturn(formatOutput(task.get("choices")));
            } else {
                logger.error("Unknown file type: " + task.get("fpath"));
            }
            if (!result) {
                String prePrompt = "# You just completed an Android code completion task:\n This is the code you need to complete:\n";
                prompt = prePrompt + prompt + "\n# This is the code you completed:\n" + task.get("choices");
                task.put("prompt", prompt + "\n# However, the code you completed may returns wrong value. Please check the code and try again.");
                this.wrongTasks.add(task);
            }
        }
    }

    public static String removeNewMethods(String code) {
        String[] targets = {"public", "private", "protected", "fun", "static", "class"};
        String[] lines = code.split("\n");
        if (lines.length > 2) {
            StringBuilder x = new StringBuilder();
            for (int i = 2; i < lines.length; i++) {
                x.append(lines[i]).append("\n");
            }
            String xString = x.toString();
            for (String target : targets) {
                if (xString.contains(target)) {
                    return code.substring(0, code.lastIndexOf(target));
                }
            }
        }
        return code;
    }
    
    private void checkBrackets () {
        for (Map<String, String> task : this.tasks) {
            String prompt = task.get("prompt");
            if (prompt.contains(indexPrompt)) {
                prompt = prompt.substring(prompt.indexOf(indexPrompt) + indexPrompt.length());
            }
            String code = formatOutput(prompt + '\n' + task.get("choices"));
            boolean result = matchBrackets(code);
            if (!result) result = matchBrackets(formatOutput(task.get("choices")));
            if (!result) {
                String prePrompt = "# You just completed an Android code completion task:\n This is the code you need to complete:\n";
                prompt = prePrompt + prompt + "\n# This is the code you completed:\n" + task.get("choices");
                task.put("prompt", prompt + "\n# However, the code you completed can not be compiled. Please check the code and try again.");
                this.wrongTasks.add(task);
            }
        }
    }

    private static boolean matchBrackets(String input) {
        Deque<Character> stack = new LinkedList<>();
        for (char bracket : input.toCharArray()) {
            if (isOpenBracket(bracket)) {
                stack.push(getClosingBracket(bracket));
            } else if (isCloseBracket(bracket)) {
                if (stack.isEmpty()) {
                    return false;
                } else {
                    stack.pop();
                }
            }
        }
        return stack.isEmpty();
    }

    private static boolean isOpenBracket(char bracket) {
        return bracket == '{' || bracket == '[' || bracket == '(';
    }

    private static boolean isCloseBracket(char bracket) {
        return bracket == '}' || bracket == ']' || bracket == ')';
    }

    private static char getClosingBracket(char openBracket) {
        switch (openBracket) {
            case '{': return '}';
            case '[': return ']';
            case '(': return ')';
            default: throw new IllegalArgumentException("Invalid bracket: " + openBracket);
        }
    }

    private static String formatOutput (String output) {
        StringBuilder result = new StringBuilder();
        for (String line : output.split("\n")) {
            if (!line.trim().startsWith("import")) {
                result.append(line).append("\n");
            }
        }
        return result.toString();
    }
}
