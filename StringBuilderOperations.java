import java.util.*;
public class StringBuilderOperations {
StringBuilder sb = new StringBuilder("helloWorld");
System.out.println(sb); // helloWorld
sb.append("Java");
System.out.println("  append: " + sb); //   append: helloWorldJava
// Insert at index
sb.insert(5, "_");
System.out.println("  insert: " + sb); //   insert: hello_WorldJava

// Delete characters (index 2 to 3)
sb.delete(2, 4); 
System.out.println("  delete: " + sb); //   delete: heo_WorldJava

// Reverse entire string
sb.reverse();
System.out.println("  reverse: " + sb);  //   reverse: avaJdlroW_oeh

// Convert to String
String finalString = sb.toString();
System.out.println("Converted to String: " + finalString); // Converted to String: avaJdlroW_oeh

// Char at index
char ch = sb.charAt(2);
System.out.println("Char at index 2: " + ch); // Char at index 2: a

// Replace char at index
sb.setCharAt(2, '#');
System.out.println("  setCharAt: " + sb); //   setCharAt: av#JdlroW_oeh

// Length
System.out.println("Length: " + sb.length()); // Length: 13

// Index of substring
System.out.println("Index of 'ava': " + sb.indexOf("dl")); // Index of 'ava': 4

// Replace substring
sb.replace(1, 4, "XYZ");
System.out.println("  replace: " + sb);  //   replace: aXYZdlroW_oeh
    }
}

