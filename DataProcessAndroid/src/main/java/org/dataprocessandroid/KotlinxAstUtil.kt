package org.callgraph

import kotlinx.ast.common.AstSource
import kotlinx.ast.common.ast.AstNode
import kotlinx.ast.common.ast.astInfoOrNull
import kotlinx.ast.common.ast.rawAstOrNull
import kotlinx.ast.common.klass.KlassDeclaration
import kotlinx.ast.grammar.kotlin.common.summary
import kotlinx.ast.grammar.kotlin.target.antlr.kotlin.KotlinGrammarAntlrKotlinParser
import java.io.File
import java.nio.file.Files
import java.util.*

class KotlinxAstUtil {
    private val cache = mutableMapOf<String, List<String>>()

    fun findMethodCall (path: String, methodName: String, callMethods: MutableList<String>, androidFlag: Boolean): Boolean {
        if (callMethods.size >= 2) return true // limit the number of call methods
        if (path.endsWith("calendar\\pro\\helpers\\IcsImporter.kt")) return true // skip this file
        var flag = true
        if (path in cache) {
            cache[path]?.forEach { addCallMethod(it, methodName, callMethods, androidFlag) }
        } else {
            if (androidFlag) {
                getAllMethodsCode(path).forEach { addCallMethod(it, methodName, callMethods, true) }
            } else {
                val kotlinCode = File(path).readText()
                val kotlinFile = KotlinGrammarAntlrKotlinParser.parseKotlinFile(AstSource.File(path))
                kotlinFile.summary(attachRawAst = false)
                .onSuccess { astList ->
                    astList.forEach { ast ->
                        methodCallVisitor(ast, path, kotlinCode, methodName, callMethods)
                    }
                }.onFailure {
                    println("Kotlinx-ast error parsing file: $path")
                    flag = false
                }
            }
        }
        return flag
    }

    private fun methodCallVisitor(ast: Any, path: String, code: String, name: String, callMethods: MutableList<String>) {
        when (ast) {
            is KlassDeclaration -> {
                if (ast.keyword != "fun") {
                    ast.children.forEach { methodCallVisitor(it, path, code, name, callMethods) }
                    return
                }
                val methodNode = ast.rawAstOrNull()?.ast as? AstNode ?: return
                val method = methodNode.children.lastOrNull { it.description == "functionBody" } ?: return
                val target = method.astInfoOrNull ?: return
                val methodCode = code.substring(target.start.index, target.stop.index)
                val methodList = cache.getOrPut(path) { mutableListOf() }.toMutableList()
                methodList.add(methodCode)
                cache[path] = methodList
                addCallMethod(methodCode, name, callMethods, false)
            }
            is AstNode -> ast.children.forEach { methodCallVisitor(it, path, code, name, callMethods) }
        }
    }

    private fun addCallMethod (methodCode: String, name: String, callMethods: MutableList<String>, androidFlag: Boolean = false) {
        if (!androidFlag && methodCode.length > 100) {
            methodCode.split("\n").forEachIndexed { index, line ->
                if (line.contains(name)) {
                    var context = line
                    if (index < methodCode.lines().size - 1) {
                       context = context + "\n" + methodCode.lines()[index + 1]
                    }
                    if (index > 0) {
                        context = methodCode.lines()[index - 1] + "\n" + context
                    }
                    callMethods.add(context)
                }
            }
        } else if (methodCode.contains(name)) {
            callMethods.add(methodCode)
        }
    }

    fun clearCache() {
        cache.clear()
    }

    fun getAllMethods (path: String): List<Array<String>> {
        val methods = mutableListOf<Array<String>>()
        val variables = mutableListOf<String>()
        if (path.endsWith("calendar\\pro\\helpers\\IcsImporter.kt")) return methods // skip this file
        val kotlinFile = KotlinGrammarAntlrKotlinParser.parseKotlinFile(AstSource.File(path))
        kotlinFile.summary(attachRawAst = false)
        .onSuccess { astList ->
            astList.forEach { ast ->
                methodVisitor(ast, path, methods, variables)
            }
        }.onFailure {
            println("Kotlinx-ast error parsing file: $path")
        }
        methods.add(variables.toTypedArray())
        return methods
    }

    private fun getAllMethodsCode (path: String): List<String> {
        val methods = mutableListOf<String>()
        val res = getAllMethods(path)
        res.forEach { methods.add(it[0]) }
        return methods
    }

    private fun methodVisitor(ast: Any, path: String, methods: MutableList<Array<String>>, variables: MutableList<String>) {
        when (ast) {
            is KlassDeclaration -> {
                if (ast.keyword != "fun") {
                    ast.children.forEach { methodVisitor(it, path, methods, variables) }
                    return
                }
                val methodNode = ast.rawAstOrNull()?.ast as? AstNode ?: return
                val method = methodNode.children.lastOrNull { it.description == "functionBody" } ?: return
                var target = method.astInfoOrNull ?: return
                val methodBody = File(path).readText().substring(target.start.index, target.stop.index)
                if (methodBody.split("\n").size <= 3) return
                if (methodBody.split("\n").size > 60) return
                if (methodBody.length > 512) return
                target = methodNode.astInfoOrNull ?: return
                val methodName = ast.identifier?.identifier ?: return
                val methodCode = File(path).readText().substring(target.start.index, target.stop.index)
                methods.add(arrayOf(methodCode, methodBody, target.start.line.toString(), target.stop.line.toString(), methodName))
            }
            is AstNode -> {
                 if (ast.description == "classBody") {
                     ast.children.filterIsInstance<KlassDeclaration>()
                        .filter { it.keyword == "var" || it.keyword == "val" }
                        .map { extractKClassContent(it.description) }
                        .filterNot { it in variables }
                        .forEach { variables.add(it) }
                }
                ast.children.forEach { methodVisitor(it, path, methods, variables) }
            }
        }
    }

    fun checkVariables (code: String): Boolean {
        val tempFile = Files.createTempFile("temp-kotlin-", ".kt").toFile()
        var flag = true
        val parameters = emptyList<String>().toMutableList()
        try {
            tempFile.writeText(code)
            val kotlinFile = KotlinGrammarAntlrKotlinParser.parseKotlinFile(AstSource.File(tempFile.absolutePath))
            kotlinFile.summary(attachRawAst = false)
                .onSuccess { astList ->
                    astList.forEach { ast ->
                        methodVariableVisitor(ast, parameters)
                    }
                }.onFailure { errors ->
                    // errors.forEach(::println)
                }
        } finally {
            tempFile.delete()
        }
        for (parameter in parameters) {
            if (code.indexOf(parameter) == code.lastIndexOf(parameter)) {
                flag = false
                break
            }
        }
        return flag
    }

    private fun methodVariableVisitor(ast: Any, parameters: MutableList<String>) {
        when (ast) {
            is KlassDeclaration -> {
                if (ast.keyword != "fun") {
                    ast.children.forEach { methodVariableVisitor(it, parameters) }
                    return
                }
                parameters.addAll(ast.parameter.map { it.identifier?.identifier.toString() })
            }
            is AstNode -> ast.children.forEach { methodVariableVisitor(it, parameters) }
        }
    }

    fun checkReturn (code: String): Boolean {
        val tempFile = Files.createTempFile("temp-kotlin-", ".kt").toFile()
        val parameters = emptyList<String>().toMutableList()
        try {
            tempFile.writeText(code)
            val kotlinFile = KotlinGrammarAntlrKotlinParser.parseKotlinFile(AstSource.File(tempFile.absolutePath))
            kotlinFile.summary(attachRawAst = false)
                .onSuccess { astList ->
                    astList.forEach { ast ->
                        methodReturnVisitor(ast, parameters)
                    }
                }.onFailure { errors ->
                    // errors.forEach(::println)
                }
        } finally {
            tempFile.delete()
        }
        return parameters.isEmpty()
    }

    private fun methodReturnVisitor(ast: Any, parameters: MutableList<String>) {
        when (ast) {
            is KlassDeclaration -> {
                if (ast.keyword != "fun") {
                    ast.children.forEach { methodReturnVisitor(it, parameters) }
                    return
                }
                if (ast.type.isNotEmpty()) {
                    parameters.addAll(ast.type.map { it.identifier })
                }
            }
            is AstNode -> ast.children.forEach { methodReturnVisitor(it, parameters) }
        }
    }

    companion object {
        @JvmStatic
        fun getImportList(source: String): List<String> {
            val importList = emptyList<String>().toMutableList()
            if (source.endsWith("calendar\\pro\\helpers\\IcsImporter.kt")) return importList // skip this file
            val kotlinFile = KotlinGrammarAntlrKotlinParser.parseKotlinFile(AstSource.File(source))
            kotlinFile.summary(attachRawAst = false)
                .onSuccess { astList ->
                    astList.forEach { ast ->
                        if (ast is AstNode && ast.description == "importList") {
                            ast.children.forEach { importAst ->
                                val importContent = extractImportContent(importAst.description)
                                if (importContent.isNotBlank()) importList.add(importContent)
                            }
                        }
                    }
                }.onFailure { errors ->
                    errors.forEach(::println)
                }
            return importList
        }

        @JvmStatic
        fun getAndroidImports(source: String): List<String> {
            val imports = emptyList<String>().toMutableList()
            getImportList(source).forEach {
                val importNames = it.split(".")
                val lastName = importNames.last().toLowerCase(Locale.getDefault())
                if (importNames.first() in listOf("android", "androidx") && !lastName.endsWith("utils")) {
                    imports.add(importNames.last())
                }
            }
            return imports
        }

        @JvmStatic
        fun extractImportContent(importStatement: String): String {
            val importRegex = "^\\s*Import\\((.*)\\)\\s*$".toRegex()
            val matchResult = importRegex.find(importStatement)
            return matchResult?.groups?.get(1)?.value ?: ""
        }

        @JvmStatic
        fun extractKClassContent(importStatement: String): String {
            val importRegex = "^\\s*KlassDeclaration\\((.*)\\)\\s*$".toRegex()
            val matchResult = importRegex.find(importStatement)
            return matchResult?.groups?.get(1)?.value ?: ""
        }
    }
}