package org.callgraph
import com.fasterxml.jackson.core.type.TypeReference
import com.fasterxml.jackson.databind.JsonNode
import com.fasterxml.jackson.databind.ObjectMapper
import kastree.ast.Node
import kastree.ast.Visitor
import kastree.ast.Writer
import kastree.ast.psi.Parser
import org.apache.logging.log4j.LogManager
import org.apache.logging.log4j.Logger
import java.io.File
import java.util.regex.Pattern

class KotlinCallFinder {
    private val logger: Logger = LogManager.getLogger("KotlinCallFinder")
    private val errorFiles = emptyList<String>().toMutableList()
    private val kotlinxAstUtil = KotlinxAstUtil()
    private val cache = mutableMapOf<String, String>()
    private val lifeCycleMap = LifeCycleMap().lifecycleMap

    fun executeCallFinding (projectSourcePaths: Array<String>, tasks: Map<String, List<Array<String>>>, fpath: String): Map<String, Array<String>> {
        var callTasks = mutableMapOf<String, Array<String>>()
//        callTasks = loadCallTasks()?.toMutableMap() ?: mutableMapOf()
        for (projectSourcePath in projectSourcePaths) {
            var repo = projectSourcePath.substring(projectSourcePath.indexOf("train" + File.separator) + 6)
            repo = repo.substring(0, repo.indexOf(File.separator))
            val path = projectSourcePath.substring(0, projectSourcePath.indexOf(repo) + repo.length)
            if (repo !in tasks) continue
            var idx = 0
            val sum = tasks.values.sumOf { it.size }
            // 遍历repo下所有kt文件找到调用点
            val results = File(projectSourcePath).walkTopDown().filter { it.isFile && it.extension == "kt" }
            tasks[repo]?.forEach { kotlinCode ->
                var callMethodsStr = ""
                if (kotlinCode[0] in cache) {
                    callMethodsStr = cache[kotlinCode[0]]!!
                } else {
                    val callMethods = findMethodCall(kotlinCode, results.toList())
                    if (callMethods.isNotEmpty()) {
                        callMethodsStr = callMethods.joinToString("\n" + "-".repeat(50) + "\n")
                    }
                    cache[kotlinCode[0]] = callMethodsStr
                }
                callTasks[kotlinCode[2]] = arrayOf(callMethodsStr, findMethodTypes(path, kotlinCode))
                if (idx % 20 == 0) logger.info("[$idx/${tasks[repo]?.size}/$sum] $repo")
                idx++
            }
            cache.clear()
            kotlinxAstUtil.clearCache()
            // Utils.writeCallJsonFile(callTasks, fpath)
        }
        return callTasks
    }

    fun extractAllMethods (path: String): List<Array<String>> {
        var methods = emptyList<Array<String>>().toMutableList()
        try {
            methods = kotlinxAstUtil.getAllMethods(path).toMutableList()
        } catch (_: Exception) { }
        return methods
    }

    private fun findMethodTypes (path: String, code: Array<String>): String {
        var methodTypes = code[0]
        val file = File(path + File.separator + code[1])
        if (file.absolutePath in errorFiles) return methodTypes
        try {
            val imports = KotlinxAstUtil.getImportList(file.path)
            for (i in imports) {
                if (code[0].contains("$i ") || code[0].contains(" $i")) {
                    methodTypes += "\nimport $i"
                }
            }
        } catch (e: Exception) {
            try {
                val node = parser.parseFile(file.readText().replace("\r\n", "\n"))
                for (i in node.imports) {
                    val target = i.names[i.names.size - 1]
                    if (code[0].contains("$target ") || code[0].contains(" $target")) {
                        methodTypes += "\n" + i.toString()
                    }
                }
            } catch (e: Exception) {
                logger.error("Error in findMethodTypes parsing file: ${file.name}")
                errorFiles.add(file.absolutePath)
            }
        }
        return methodTypes
    }

    private fun extractCallName (input: String): String? {
        val startIndex = input.indexOf("name=")
        if (startIndex != -1) {
            val endIndex = input.indexOf(')', startIndex)
            if (endIndex != -1) {
                return input.substring(startIndex + 5, endIndex)
            }
        }
        return null
    }

    private fun findMethodCall (code: Array<String>, results: List<File>): List<String> {
        var methodName = code[3].ifEmpty { getMethodName(code[0]) }
        val callMethods = emptyList<String>().toMutableList()
        if (methodName == null) return callMethods
        var files = results
        val flag = (code[4] == "android") && this.lifeCycleMap.containsKey(methodName)
        if (flag) {
            files = results.filter { it.absolutePath.contains(code[1]) }
            methodName = this.lifeCycleMap[methodName]!!  // convert to relative lifecycle method
        }
        files.forEach { file ->
            try {
                if (file.absolutePath in errorFiles) return@forEach
                val node = parser.parseFile(file.readText().replace("\r\n", "\n"))
                if (flag) {
                    Visitor.visit(node) { v, _ ->
                        if(v is Node.Decl.Func) {
                            if (v.body != null && v.name.toString() == methodName) {
                                callMethods.add(Writer.write(v))
                            }
                        }
                    }
                } else {
                    Visitor.visit(node) { v, p ->
                        if(v is Node.Expr.Call) {
                            val calledMethod = extractCallName(v.expr.toString())
                            if (calledMethod != null && calledMethod == methodName) {
                                if (Writer.write(p).length > 100) callMethods.add(Writer.write(v))
                                else callMethods.add(Writer.write(p))
                            }
                        }
                    }
                }
            } catch (e: Exception) {
                try{
                    if (file.absolutePath in errorFiles) return@forEach
                    if (!kotlinxAstUtil.findMethodCall(file.path, methodName, callMethods, flag)) errorFiles.add(file.absolutePath)
                } catch (e: Exception) {
                    logger.error("Error in findMethodCall parsing file: ${file.name}")
                    errorFiles.add(file.absolutePath)
                }
            }
        }
        return callMethods.distinct()
    }

    fun checkKotlinReturn (input: String): Boolean {
        val code = input.replace("\r\n", "\n")
        var checkFlag = true
        try {
            val node = parser.parseFile(code)
            Visitor.visit(node) { v, _ ->
                if(v is Node.Decl.Func) {
                    if (v.body != null) {
                        checkFlag = (v.type == null) xor hasReturnStmtWithValue(code)
                    }
                }
            }
        } catch (e: Exception) {
            try {
                val noReturn = kotlinxAstUtil.checkReturn(code)
                checkFlag = noReturn xor hasReturnStmtWithValue(code)
            } catch (e: Exception) {
                // logger.error("Error parsing code: $code")
                return false
            }
        }
        return checkFlag
    }

    private fun hasReturnStmtWithValue(functionBody: String): Boolean {
        val returnTypeRegex = Regex("""\breturn\b\s+\w+""")
        return returnTypeRegex.containsMatchIn(functionBody.substring(functionBody.indexOf("fun")).trimIndent().replace("return\n", ""))
    }

    fun checkKotlinVariables (code: String): Boolean {
        var checkFlag = true
        val variableNames = emptyList<String>().toMutableList()
        try {
            val node = parser.parseFile(code.replace("\r\n", "\n"))
            Visitor.visit(node) { v, _ ->
                if(v is Node.Decl.Property) {
                    v.vars.forEach{ one ->
                        if (one != null) {
                            variableNames.add(one.name)
                        }
                    }
                }
            }
            for (i in variableNames) {
                if (code.indexOf(i) == code.lastIndexOf(i)) {
                    checkFlag = false
                    break
                }
            }
        } catch (e: Exception) {
            try {
                checkFlag = kotlinxAstUtil.checkVariables(code)
            } catch (e: Exception) {
                // logger.error("Error parsing code: $code")
            }
        }
        return checkFlag
    }

    private fun loadCallTasks(): Map<String, Array<String>>? {
        val jsonNode: JsonNode? = readJson("new_call_tasks_train.json")
        require(jsonNode != null && jsonNode.isObject) { "JsonNode is null or not an object" }
        val objectMapper = ObjectMapper()
        return try {
            objectMapper.readValue(jsonNode.traverse(), object : TypeReference<Map<String, Array<String>>>() {})
        } catch (e: Exception) {
            logger.error(e)
            null
        }
    }

    private fun readJson(filePath: String): JsonNode? {
        return try {
            val objectMapper = ObjectMapper()
            val file = File(filePath)
            objectMapper.readTree(file)
        } catch (e: Exception) {
            logger.error(e)
            null
        }
    }

    companion object {
        private val parser = Parser()

        @JvmStatic
        fun findAndroidImports (path: String): List<String> {
            var imports = emptyList<String>().toMutableList()
            val file = File(path)
            try {
                val node = parser.parseFile(file.readText().replace("\r\n", "\n"))
                for (i in node.imports) {
                    val names = i.names
                    if (names[0].startsWith("android") || names.joinToString(".").startsWith("com.google.android")) {
                        if (!names[names.size - 1].toLowerCase().endsWith("utils")) {
                            imports.add(names[names.size - 1])
                        }
                    }
                }
            } catch (e: Exception) {
                try {
                    imports = KotlinxAstUtil.getAndroidImports(path).toMutableList()
                } catch (e: Exception) {
                    println("Error parsing file: ${file.name}")
                }
            }
            return imports.distinct()
        }

        @JvmStatic
        fun getAllImports (path: String): List<String> {
            var imports = emptyList<String>().toMutableList()
            val file = File(path)
            try {
                val node = parser.parseFile(file.readText().replace("\r\n", "\n"))
                for (i in node.imports) {
                    imports.add(i.names.joinToString("."))
                }
            } catch (e: Exception) {
                try {
                    imports = KotlinxAstUtil.getImportList(path).toMutableList()
                } catch (e: Exception) {
                    println("Error parsing file: ${file.name}")
                }
            }
            return imports.distinct()
        }


        @JvmStatic
        fun getMethodName (input: String): String? {
            val code = input.substring(input.indexOf("fun")).trim()
            var methodName: String? = null
            try {
                val file = parser.parseFile(code.replace("\r\n", "\n"))
                Visitor.visit(file) { v, _ ->
                    if(v is Node.Decl.Func) {
                        if (v.name != null && v.name.toString().length > 3) methodName = v.name.toString()
                    }
                }
            } catch (e: Exception) {
                val pattern = Pattern.compile("fun\\s+([a-zA-Z0-9_]+)\\s*\\(")
                val matcher = pattern.matcher(code)
                if (matcher.find()) {
                    methodName = matcher.group(1)
                } else {
                    methodName = code.substring(0, code.indexOf("(")).split(" ").last()
                    if (methodName!!.contains(".")) methodName = methodName!!.substring(methodName!!.indexOf(".") + 1)
                    // println("Error extract method name: $code")
                }
            }
            return methodName
        }
    }

}