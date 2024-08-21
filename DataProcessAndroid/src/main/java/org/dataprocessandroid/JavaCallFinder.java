package org.callgraph;

import com.github.javaparser.JavaParser;
import com.github.javaparser.ParseResult;
import com.github.javaparser.ParserConfiguration;
import com.github.javaparser.StaticJavaParser;
import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.body.ConstructorDeclaration;
import com.github.javaparser.ast.body.FieldDeclaration;
import com.github.javaparser.ast.body.MethodDeclaration;
import com.github.javaparser.ast.expr.MethodCallExpr;
import com.github.javaparser.ast.expr.VariableDeclarationExpr;
import com.github.javaparser.ast.stmt.ReturnStmt;
import com.github.javaparser.ast.visitor.VoidVisitorAdapter;
import com.github.javaparser.symbolsolver.JavaSymbolSolver;
import com.github.javaparser.symbolsolver.resolution.typesolvers.CombinedTypeSolver;
import com.github.javaparser.symbolsolver.resolution.typesolvers.ReflectionTypeSolver;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class JavaCallFinder {
    private static final Logger logger = LogManager.getLogger(JavaCallFinder.class);
    private final String[] projectSourcePaths;
    private final List<String> errorFiles = new ArrayList<>();
    private final Map<String, List<String[]>> tasks;
    private final JavaParser javaParser;
    private final Map<String, CompilationUnit> cache;
    private final Map<String, String> lifeCycleMap;

    public JavaCallFinder() {
        this.projectSourcePaths = new String[]{};
        this.tasks = new HashMap<>();
        CombinedTypeSolver combinedSolver = new CombinedTypeSolver();
        combinedSolver.add(new ReflectionTypeSolver());
        ParserConfiguration parserConfiguration = new ParserConfiguration();
        parserConfiguration.setLanguageLevel(ParserConfiguration.LanguageLevel.JAVA_17);
        parserConfiguration.setSymbolResolver(new JavaSymbolSolver(combinedSolver));
        this.javaParser = new JavaParser(parserConfiguration);
        this.cache = new HashMap<>();
        this.lifeCycleMap = new LifeCycleMap().getLifecycleMap();
    }

    public JavaCallFinder(String[] projectSourcePaths, Map<String, List<String[]>> tasks) {
        this.projectSourcePaths = projectSourcePaths;
        this.tasks = tasks;
        this.javaParser = new JavaParser();
        this.cache = new HashMap<>();
        this.lifeCycleMap = new LifeCycleMap().getLifecycleMap();
    }

    public Map<String, String[]> executeCallFinding () {
        Map<String, String[]> callTasks = new HashMap<>();
        for (String projectSourcePath : projectSourcePaths) {
            String repo = projectSourcePath.substring(projectSourcePath.indexOf("train" + File.separator) + 6);
            repo = repo.substring(0, repo.indexOf(File.separator));
            if (!tasks.containsKey(repo)) continue;
            List<String[]> methods = tasks.get(repo);
            int sum = 0;
            for (List<String[]> task: tasks.values()) {
                sum += task.size();
            }
            int idx = 0;
            List<String> results = new ArrayList<>();
            try (Stream<Path> walk = Files.walk(Paths.get(projectSourcePath))) {
                results = walk.filter(Files::isRegularFile).map(Path::toString)
                        .filter(string -> string.endsWith(".java")).collect(Collectors.toList());
            } catch (IOException e) {
                logger.error("exception message", e);
            }
            for (String[] method : methods) {
                String methodStr;
                StringBuilder sb = new StringBuilder();
                for (String callingMethod : findMethodCall(results, method)) {
                    sb.append(callingMethod).append("\n").append("--------------------------------------------------\n");
                }
                methodStr = sb.toString();
                callTasks.put(method[2], new String[]{methodStr, findMethodTypes(projectSourcePath.substring(0, projectSourcePath.indexOf(repo) + repo.length()), method)});
                idx++;
                if (idx % 20 == 0) {
                    logger.info(repo + ": " + idx + "/" + methods.size() + ", total: " + sum);
                }
            }
            cache.clear();
            errorFiles.clear();
        }
        return callTasks;
    }

    public List<String[]> extractAllMethods (String path) {
        List<String[]> methods = new ArrayList<>();
        try {
            String fileContent = Utils.readFile(path);
            ParseResult<CompilationUnit> parseResult = javaParser.parse(fileContent);
            CompilationUnit cu = parseResult.getResult().orElseThrow(() -> new RuntimeException(parseResult.getProblems().get(0).getMessage()));
            List<MethodDeclaration> methodDeclarations = cu.findAll(MethodDeclaration.class);
            for (MethodDeclaration methodDeclaration : methodDeclarations) {
                if (!methodDeclaration.getBody().isPresent()) continue;
                if (!methodDeclaration.getRange().isPresent()) continue;
                if (methodDeclaration.getBody().toString().split("\n").length <= 3) continue;
                if (methodDeclaration.getBody().toString().split("\n").length > 60) continue;
                if (methodDeclaration.getBody().toString().length() > 512) continue;
                String begin = String.valueOf(methodDeclaration.getRange().get().begin.line);
                String end = String.valueOf(methodDeclaration.getRange().get().end.line);
                String code = methodDeclaration.toString();
                methods.add(new String[]{code, methodDeclaration.getBody().map(Object::toString).orElse(""), begin, end, methodDeclaration.getNameAsString()});
            }
            List<FieldDeclaration> fieldDeclarations = cu.findAll(FieldDeclaration.class);
            List<String> fields = new ArrayList<>();
            for (FieldDeclaration fieldDeclaration : fieldDeclarations) {
                if (fieldDeclaration.toString().startsWith("//")) continue;
                fields.add(fieldDeclaration.toString());
            }
            methods.add(fields.toArray(new String[0]));
        } catch (Exception e) {
            logger.error("exception message", e);
        }
        return methods;
    }

    public static String getMethodName (String method) {
        String methodName;
        try {
            CompilationUnit cu = StaticJavaParser.parse(fakeClass(method));
            List<MethodDeclaration> methodDeclarations = cu.findAll(MethodDeclaration.class);
            if (!methodDeclarations.isEmpty()) {
                methodName = methodDeclarations.get(0).getNameAsString();
            } else {
                methodName = cu.findAll(ConstructorDeclaration.class).get(0).getNameAsString();
            }
        } catch (Exception e) {
            Pattern pattern = Pattern.compile("\\b\\w+\\s+(\\w+)\\s*\\(");
            Matcher matcher = pattern.matcher(method);

            if (matcher.find()) {
                return matcher.group(1);
            } else {
                logger.error("exception message", e);
                logger.error("method: " + method);
            }
            return "";
        }
        return methodName;
    }

    public static String fakeClass (String method) {
        return "public class FakeClass { " + method + " }";
    }

    private String findMethodTypes (String projectSourcePath, String[] method) {
        String filepath = projectSourcePath + File.separator + method[1];
        if (this.errorFiles.contains(filepath)) return method[0];
        StringBuilder sb = new StringBuilder();
        try {
            CompilationUnit cu;
            if (this.cache.containsKey(filepath)) {
                cu = this.cache.get(filepath);
            } else {
                String fileContent = Utils.readFile(filepath);
                ParseResult<CompilationUnit> parseResult = javaParser.parse(fileContent);
                cu = parseResult.getResult().orElseThrow(() -> {
                    this.errorFiles.add(filepath);
                    return new RuntimeException(parseResult.getProblems().get(0).getMessage());
                });
                this.cache.put(filepath, cu);
            }
            assert cu != null;
            cu.getImports().forEach(importDeclaration -> {
                String importName = importDeclaration.getNameAsString();
                String target = importName.substring(importName.lastIndexOf(".") + 1);
                if (method[0].contains(target + " ") || method[0].contains(" " + target)) {
                    sb.append(importDeclaration);
                }
            });
        } catch (Exception e) {
            logger.error("exception message", e);
        }
        sb.append(method[0]);
        return sb.toString();
    }

    private List<String> findMethodCall (List<String> results, String[] method) {
        List<String> callingMethods = new ArrayList<>();
        try {
            String methodName = method[3].isEmpty() ? getMethodName(method[0]) : method[3];
            if (methodName.isEmpty()) return callingMethods;
            boolean flag = this.lifeCycleMap.containsKey(methodName) && method[4].equals("android");
            if (flag) {
                results = results.stream().filter(file -> file.contains(method[1])).collect(Collectors.toList());
            }
            for (String file: results) {
                if (this.errorFiles.contains(file)) continue;
                try {
                    CompilationUnit cu;
                    if (this.cache.containsKey(file)) {
                        cu = this.cache.get(file);
                    } else {
                        String fileContent = Utils.readFile(file);
                        ParseResult<CompilationUnit> parseResult = javaParser.parse(fileContent);
                        cu = parseResult.getResult().orElseThrow(() -> {
                            this.errorFiles.add(file);
                            return new RuntimeException(parseResult.getProblems().get(0).getMessage());
                        });
                        this.cache.put(file, cu);
                    }
                    assert cu != null;
                    if (flag) {  // android lifecycle method
                        List<MethodDeclaration> methodDeclarations = cu.findAll(MethodDeclaration.class);
                        methodDeclarations.stream()
                            .filter(methodDeclaration -> methodDeclaration.getNameAsString().equals(this.lifeCycleMap.get(methodName)))
                            .filter(methodDeclaration -> methodDeclaration.getBody().isPresent())
                            .forEach(methodDeclaration -> callingMethods.add(methodDeclaration.toString()));
                    } else {
                        MethodCallVisitor methodCallVisitor = new MethodCallVisitor(methodName, callingMethods);
                        cu.accept(methodCallVisitor, null);
                    }
                } catch (Exception e) {
                    logger.error("exception message", e);
                }
            }
        } catch (Exception e) {
            logger.error("exception message", e);
        }
        return callingMethods;
    }

    private static class MethodCallVisitor extends VoidVisitorAdapter<Void> {
        private final String targetMethodName;
        private final List<String> callingMethods;
        MethodCallVisitor(String targetMethodName, List<String> callingMethods) {
            this.targetMethodName = targetMethodName;
            this.callingMethods = callingMethods;
        }

        @Override
        public void visit(MethodCallExpr methodCallExpr, Void arg) {
            super.visit(methodCallExpr, arg);
            if (methodCallExpr.getNameAsString().equals(this.targetMethodName)) {
                methodCallExpr.findAncestor(MethodDeclaration.class).ifPresent(
                    callingMethod -> {
                        if (this.callingMethods.size() < 2){
                            this.callingMethods.add(callingMethod.toString());
                        }
                    }
                );
            }
        }
    }

    public static List<String> findAndroidImports (String filePath) {
        String code = Utils.readFile(filePath);
        List<String> imports = new ArrayList<>();
        try {
            CompilationUnit cu = StaticJavaParser.parse(code);
            cu.getImports().forEach(importDeclaration -> {
                String importName = importDeclaration.getNameAsString();
                if (importName.startsWith("android") || importName.startsWith("com.google.android.material")) {
                    imports.add(importName.substring(importName.lastIndexOf(".") + 1));
                }
            });
        } catch (Exception e) {
            logger.error("exception message", e);
        }
        return imports;
    }

    public static List<String> getAllImports (String filePath) {
        String code = Utils.readFile(filePath);
        List<String> imports = new ArrayList<>();
        try {
            CompilationUnit cu = StaticJavaParser.parse(code);
            cu.getImports().forEach(importDeclaration -> imports.add(importDeclaration.getNameAsString()));
        } catch (Exception e) {
            logger.error("exception message", e);
        }
        return imports;
    }

    public boolean checkJavaReturn (String code) {
        try {
            ParseResult<CompilationUnit> parseResult = javaParser.parse(fakeClass(code));
            CompilationUnit cu = parseResult.getResult().orElseThrow(() -> new RuntimeException(parseResult.getProblems().get(0).getMessage()));
            List<MethodDeclaration> methodDeclarations = cu.findAll(MethodDeclaration.class);
            for (MethodDeclaration methodDeclaration : methodDeclarations) {
                String type = methodDeclaration.getType().asString().toLowerCase();
                List<ReturnStmt> returnStatements = methodDeclaration.findAll(ReturnStmt.class);
                for (ReturnStmt returnStmt : returnStatements) {
                    // void return check
                    if (returnStmt.getExpression().isPresent() && Objects.equals(type, "void")) continue;
                    if (returnStmt.getExpression().toString().contains("Optional.empty") && Objects.equals(type, "void")) continue;
                    // in-project type definition return check
                    if (returnStmt.getExpression().toString().toLowerCase().contains(type)) continue;
                    // normal return check
                    String realType = returnStmt.getExpression().get().calculateResolvedType().describe();
                    if (!(Objects.equals(type, realType) || realType.equals("null"))) {
                        return false;
                    }
                }
            }
        } catch (Exception e) {
            // logger.error("exception message", e);
            return false;
        }
        return true;
    }

    private static class VariablesVisitor extends VoidVisitorAdapter<Void> {
        private final List<String> variables;

        VariablesVisitor(List<String> variables) {
            this.variables = variables;
        }

        @Override
        public void visit(VariableDeclarationExpr n, Void arg) {
            super.visit(n, arg);
            n.getVariables().forEach(v -> this.variables.add(v.getNameAsString()));
        }
    }

    public boolean checkJavaVariables (String code) {
        List<String> variables = new ArrayList<>();
        try {
            ParseResult<CompilationUnit> parseResult = javaParser.parse(fakeClass(code));
            CompilationUnit cu = parseResult.getResult().orElseThrow(() -> new RuntimeException(parseResult.getProblems().get(0).getMessage()));
            VariablesVisitor variablesVisitor = new VariablesVisitor(variables);
            cu.accept(variablesVisitor, null);
        } catch (Exception e) {
            // logger.error("exception message", e);
        }
        for (String variable : variables) {
            if (code.indexOf(variable) == code.lastIndexOf(variable)) {
                return false;
            }
        }
        return true;
    }
}
