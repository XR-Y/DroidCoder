package org.callgraph;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.dom4j.Document;
import org.dom4j.DocumentException;
import org.dom4j.Element;
import org.dom4j.Node;
import org.dom4j.io.SAXReader;

import java.io.File;
import java.io.IOException;
import java.nio.file.*;
import java.nio.file.attribute.BasicFileAttributes;
import java.util.*;

public class AndroidManifestParser {
    private static final Logger logger = LogManager.getLogger(Utils.class);
    private final String baseDir;

    public AndroidManifestParser (String baseDir) {
        this.baseDir = baseDir;
    }
    public List<Map<String, List<String>>> execute() {
        SAXReader reader = new SAXReader();
        Map<String, List<String>> androidMethodsMap = new HashMap<>();
        Map<String, List<String>> methodActionMap = new HashMap<>();
        try {
            List<String> manifestFiles = findAndroidManifestFiles(this.baseDir);
            String[] targets = new String[]{"activity", "service", "receiver", "provider", "activity-alias"};
            for (String manifestFile : manifestFiles) {
                String repoName = manifestFile.substring(manifestFile.indexOf(this.baseDir) + this.baseDir.length() + 1);
                repoName = repoName.substring(0, repoName.indexOf(File.separator));
                List<String> components = new ArrayList<>();
                Document document = reader.read(new File(manifestFile));
                Element root = document.getRootElement();
                for (String target : targets) {
                    List<Node> elements = root.selectNodes("//" + target); // Xpath
                    for (Node element : elements) {
                        assert element instanceof Element;
                        String name = ((Element) element).attributeValue("name");
                        if (name != null) {
                            if (name.startsWith(".")) name = name.substring(1);
                            name = name.replace(".", File.separator);
                            if (!components.contains(name)) components.add(name);
                        }
                        List<Node> actions = element.selectNodes("intent-filter/action/@android:name");
                        for (Node action : actions) {
                            String actionName = action.getText();
                            if (methodActionMap.containsKey(name)) {
                                methodActionMap.get(name).add(actionName);
                            } else {
                                methodActionMap.put(name, new ArrayList<>(Collections.singletonList(actionName)));
                            }
                        }
                    }
                }
                if (androidMethodsMap.containsKey(repoName)) {
                    androidMethodsMap.get(repoName).addAll(components);
                } else {
                    androidMethodsMap.put(repoName, components);
                }
                System.out.println(manifestFile + ": Finished");
            }
        } catch (DocumentException e) {
            logger.error("exception message", e);
        }
        return Arrays.asList(androidMethodsMap, methodActionMap);
    }

    public void dumpAllActions() {
        SAXReader reader = new SAXReader();
        Map<String, List<String>> methodActionMap = new HashMap<>();
        List<String> skipKeywords = Arrays.asList("top", "deprecated");
        try {
            List<String> manifestFiles = findAndroidManifestFiles(this.baseDir.substring(0, this.baseDir.indexOf("datasets")));
            String[] targets = new String[]{"activity", "service", "receiver", "provider", "activity-alias"};
            for (String manifestFile : manifestFiles) {
                if (skipKeywords.stream().anyMatch(manifestFile::contains)) continue;
                Document document = reader.read(new File(manifestFile));
                Element root = document.getRootElement();
                for (String target : targets) {
                    List<Node> elements = root.selectNodes("//" + target); // Xpath
                    for (Node element : elements) {
                        assert element instanceof Element;
                        String name = ((Element) element).attributeValue("name");
                        if (name != null) {
                            if (name.startsWith(".")) name = name.substring(1);
                            name = name.replace(".", File.separator);
                        }
                        List<Node> actions = element.selectNodes("intent-filter/action/@android:name");
                        for (Node action : actions) {
                            String actionName = action.getText();
                            if (methodActionMap.containsKey(name)) {
                                methodActionMap.get(name).add(actionName);
                            } else {
                                methodActionMap.put(name, new ArrayList<>(Collections.singletonList(actionName)));
                            }
                        }
                    }
                }
                System.out.println(manifestFile + ": Finished");
            }
        } catch (DocumentException e) {
            logger.error("exception message", e);
        }
        Utils.saveActionsMap(methodActionMap, "component_action_map.json");
    }

    public void collectPermissions() {
        SAXReader reader = new SAXReader();
        Map<String, List<String>> usesPermissionsMap = new HashMap<>();
        Map<String, Map<String, Integer>> countPermissionsMap = new HashMap<>();
        List<String> skipKeywords = Arrays.asList("top", "deprecated");
        try {
            String dir = this.baseDir.substring(0, this.baseDir.indexOf("datasets"));
            List<String> manifestFiles = findAndroidManifestFiles(dir);
            for (String manifestFile : manifestFiles) {
                if (skipKeywords.stream().anyMatch(manifestFile::contains)) continue;
                String prefix = manifestFile.contains("train") ? "train" : "test";
                String repoName = manifestFile.substring(manifestFile.indexOf(prefix) + prefix.length() + 1);
                repoName = repoName.substring(0, repoName.indexOf(File.separator));
                String typeName = manifestFile.substring(0, manifestFile.indexOf(prefix) - 1);
                typeName = typeName.substring(typeName.lastIndexOf(File.separator) + 1);
                Map<String, Integer> permissionMap = countPermissionsMap.computeIfAbsent(typeName, k -> new HashMap<>());
                Document document = reader.read(new File(manifestFile));
                Element root = document.getRootElement();
                List<Node> permissions = root.selectNodes("//uses-permission/@android:name");
                for (Node permission : permissions) {
                    String permissionName = permission.getText();
                    if (usesPermissionsMap.containsKey(repoName)) {
                        usesPermissionsMap.get(repoName).add(permissionName);
                    } else {
                        usesPermissionsMap.put(repoName, new ArrayList<>(Collections.singletonList(permissionName)));
                    }
                    permissionMap.put(permissionName, permissionMap.getOrDefault(permissionName, 0) + 1);
                }
                System.out.println(manifestFile + ": Finished");
            }
        } catch (DocumentException e) {
            logger.error("exception message", e);
        }
        for (String repo : usesPermissionsMap.keySet()) {
            System.out.println(repo + ": " + usesPermissionsMap.get(repo));
        }
        for (String typeName : countPermissionsMap.keySet()) {
            System.out.println(typeName + ": ");
            countPermissionsMap.get(typeName).entrySet().removeIf(entry -> entry.getValue() == 1);
            List<Map.Entry<String, Integer>> sortedList = new ArrayList<>(countPermissionsMap.get(typeName).entrySet());
            sortedList.sort(Map.Entry.comparingByValue(Comparator.reverseOrder()));
            for (Map.Entry<String, Integer> entry : sortedList) {
                System.out.println(entry.getKey() + " : " + entry.getValue());
            }
        }
    }

    private static class AndroidManifestVisitor extends SimpleFileVisitor<Path> {
        private final List<String> manifestFiles;

        public AndroidManifestVisitor(List<String> manifestFiles) {
            this.manifestFiles = manifestFiles;
        }

        @Override
        public FileVisitResult visitFile(Path file, BasicFileAttributes attrs) {
            if (file.getFileName().toString().equals("AndroidManifest.xml")) {
                manifestFiles.add(file.toAbsolutePath().toString());
            }
            return FileVisitResult.CONTINUE;
        }

        @Override
        public FileVisitResult visitFileFailed(Path file, IOException exc) {
            System.err.println("Failed to access: " + file.toString() + " due to " + exc.getMessage());
            return FileVisitResult.CONTINUE;
        }
    }

    public static List<String> findAndroidManifestFiles(String directoryPath) {
        List<String> manifestFiles = new ArrayList<>();
        Path startPath = Paths.get(directoryPath);

        try {
            Files.walkFileTree(startPath, new AndroidManifestVisitor(manifestFiles));
        } catch (IOException e) {
            System.err.println("Error walking through the directory: " + e.getMessage());
        }
        return manifestFiles;
    }
}

