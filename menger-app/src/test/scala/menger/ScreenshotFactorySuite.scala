package menger

import menger.engines.ScreenshotFactory
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class ScreenshotFactorySuite extends AnyFlatSpec with Matchers:

  "sanitizeFileName" should "preserve normal filenames unchanged" in:
    ScreenshotFactory.sanitizeFileName("normal_file.png") should be("normal_file.png")

  it should "preserve alphanumeric filenames with allowed characters" in:
    ScreenshotFactory.sanitizeFileName("test123.png") should be("test123.png")
    ScreenshotFactory.sanitizeFileName("my-awesome_screenshot.png") should be("my-awesome_screenshot.png")

  it should "add png extension to files without extension" in:
    ScreenshotFactory.sanitizeFileName("image") should be("image.png")

  it should "add png extension to filenames with numbers" in:
    ScreenshotFactory.sanitizeFileName("test123") should be("test123.png")

  it should "preserve existing PNG extension case-sensitively" in:
    ScreenshotFactory.sanitizeFileName("image.PNG") should be("image.PNG")

  it should "preserve existing png extension" in:
    ScreenshotFactory.sanitizeFileName("test.png") should be("test.png")

  it should "prevent simple path traversal" in:
    ScreenshotFactory.sanitizeFileName("../../../etc/passwd") should be("......etcpasswd.png")

  it should "prevent Windows path traversal" in:
    ScreenshotFactory.sanitizeFileName("..\\..\\Windows\\System32\\config") should be("....WindowsSystem32config.png")

  it should "prevent current directory path traversal" in:
    ScreenshotFactory.sanitizeFileName("./../../sensitive_file.txt") should be(".....sensitive_file.txt.png")

  it should "remove script injection attempts" in:
    ScreenshotFactory.sanitizeFileName("file<script>alert()</script>.png") should be("filescriptalertscript.png")

  it should "remove pipe and question mark characters" in:
    ScreenshotFactory.sanitizeFileName("test|file?.png") should be("testfile.png")

  it should "remove multiple dangerous characters" in:
    ScreenshotFactory.sanitizeFileName("file*with:weird|chars?.png") should be("filewithweirdchars.png")

  it should "sanitize Windows absolute path" in:
    ScreenshotFactory.sanitizeFileName("C:\\Windows\\System32\\file.txt") should be("CWindowsSystem32file.txt.png")

  it should "handle complex attack with multiple patterns" in:
    ScreenshotFactory.sanitizeFileName("../../../<script>../../file|name*.png") should be("......script....filename.png")

  it should "handle Windows path with dangerous characters" in:
    ScreenshotFactory.sanitizeFileName("..\\..\\..\\test<>?*|file.png") should be("......testfile.png")

  it should "preserve dots in middle of filename" in:
    ScreenshotFactory.sanitizeFileName("version.1.2.3.png") should be("version.1.2.3.png")

  it should "preserve config file dots and add png extension" in:
    ScreenshotFactory.sanitizeFileName("my.config.file") should be("my.config.file.png")

  it should "handle numeric only filename" in:
    ScreenshotFactory.sanitizeFileName("123") should be("123.png")

  it should "handle underscore only filename" in:
    ScreenshotFactory.sanitizeFileName("_") should be("_.png")

  it should "handle dash only filename" in:
    ScreenshotFactory.sanitizeFileName("---") should be("---.png")

  it should "throw exception for empty input" in:
    an[IllegalArgumentException] should be thrownBy ScreenshotFactory.sanitizeFileName("")

  it should "throw exception for special characters only" in:
    an[IllegalArgumentException] should be thrownBy ScreenshotFactory.sanitizeFileName("!@#$%^&*()")

  it should "throw exception for pipe characters only" in:
    an[IllegalArgumentException] should be thrownBy ScreenshotFactory.sanitizeFileName("||||||||")

  it should "throw exception for question marks only" in:
    an[IllegalArgumentException] should be thrownBy ScreenshotFactory.sanitizeFileName("????????")

  it should "preserve unicode characters" in:
    ScreenshotFactory.sanitizeFileName("file_ñàmé.png") should be("file_ñàmé.png")

  it should "preserve Chinese characters and keep extension" in:
    ScreenshotFactory.sanitizeFileName("测试文件.png") should be("测试文件.png")

  it should "handle very long filenames" in:
    val longName = "a" * 200 + ".png"
    val result = ScreenshotFactory.sanitizeFileName(longName)
    result should endWith(".png")
    result.length should be(204)

  it should "detect PNG extension case insensitively" in:
    ScreenshotFactory.sanitizeFileName("test.Png") should be("test.Png")

  "saveScreenshot security" should "sanitize path traversal before file creation" in:
    val input = "../../../etc/passwd"
    val sanitized = ScreenshotFactory.sanitizeFileName(input)
    sanitized should not contain "../"
    sanitized should be("......etcpasswd.png")

  it should "sanitize Windows paths before file creation" in:
    val input = "..\\..\\Windows\\System32\\config"
    val sanitized = ScreenshotFactory.sanitizeFileName(input)
    sanitized should not contain "\\"
    sanitized should be("....WindowsSystem32config.png")

  it should "remove all dangerous characters" in:
    val input = "file|with*dangerous?chars<>.png"
    val sanitized = ScreenshotFactory.sanitizeFileName(input)
    sanitized should not contain "|"
    sanitized should not contain "*"
    sanitized should not contain "?"
    sanitized should not contain "<"
    sanitized should not contain ">"

  it should "prevent directory traversal patterns" in:
    // These inputs contain only dots and slashes, so they become only dots after filtering
    // Since dots are allowed, these won't throw exceptions, but they'll be harmless
    val result1 = ScreenshotFactory.sanitizeFileName("../../../")
    result1 should be(".......png")

    val result2 = ScreenshotFactory.sanitizeFileName("..\\..\\")
    result2 should be(".....png")

  it should "create safe filename regex pattern" in:
    val dangerousInput = "normal/../../../etc/passwd"
    val sanitized = ScreenshotFactory.sanitizeFileName(dangerousInput)
    sanitized should fullyMatch regex "[a-zA-Z0-9._-]+\\.png"