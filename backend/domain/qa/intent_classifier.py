"""
Local intent classifier — deep research vs quick ask.

Algorithm: sklearn TF-IDF (char_wb n-gram) + Logistic Regression (primary)
           or pure-Python log-odds Naïve Bayes (fallback if sklearn absent).

Model is serialised to CODESPECTRA_DATA_DIR/classifier_model.pkl after every
training run and reloaded on startup — no retraining needed between restarts.
"""
from __future__ import annotations

import json
import math
import os
import pickle
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Final


# ---------------------------------------------------------------------------
# Training dataset
# Format: (text, is_deep_research: bool)
#
# deep_research = True   →  tracing calls, blast radius, dependency mapping,
#                            multi-hop flow analysis, exhaustive search requests
# deep_research = False  →  factual, definitional, locational, single-file questions
# ---------------------------------------------------------------------------

_DATASET: Final[list[tuple[str, bool]]] = [
    # ── English — deep research ──────────────────────────────────────────────
    ("trace the call chain from auth middleware", True),
    ("trace the execution path of this request", True),
    ("who calls this function", True),
    ("what calls this method", True),
    ("trace where this value comes from", True),
    ("trace the data flow through the pipeline", True),
    ("map out all callers of this function", True),
    ("what is the blast radius if I change this file", True),
    ("what breaks if I modify this module", True),
    ("what are the transitive dependencies of this file", True),
    ("trace the full dependency graph of this module", True),
    ("follow the execution from the entry point", True),
    ("follow the request path through the system", True),
    ("map the entire data flow", True),
    ("trace how data flows from the API to the database", True),
    ("deep research on how auth works end to end", True),
    ("give me a thorough analysis of the call graph", True),
    ("exhaustive analysis of how this module is used", True),
    ("find all files that depend on this module", True),
    ("find every place this function is called from", True),
    ("investigate the impact of removing this file", True),
    ("search very carefully through the codebase", True),
    ("search thoroughly to understand the full flow", True),
    ("dig deep into how this system works", True),
    ("trace through the entire authentication flow", True),
    ("walk through every step of the request lifecycle", True),
    ("map all the dependencies and transitive imports", True),
    ("comprehensive analysis of the event handling chain", True),
    ("trace the chain of responsibility pattern here", True),
    ("find all the files involved in this feature end to end", True),

    # ── English — quick ask ──────────────────────────────────────────────────
    ("what does this function do", False),
    ("explain this class", False),
    ("what is the purpose of this file", False),
    ("where is the auth middleware defined", False),
    ("what framework does the backend use", False),
    ("what tech stack is this project", False),
    ("how does this function work", False),
    ("what does this method return", False),
    ("what parameters does this accept", False),
    ("find the UserService class", False),
    ("show me the database schema", False),
    ("what language is the backend written in", False),
    ("what does this variable represent", False),
    ("explain the purpose of this module", False),
    ("where is the entry point", False),
    ("list all the API routes", False),
    ("what does this regex match", False),
    ("what does the constructor do", False),
    ("describe what this interface does", False),
    ("find all the test files", False),
    ("what does this config option do", False),
    ("where is the database connection set up", False),
    ("what are the environment variables", False),
    ("what does this error message mean", False),
    ("show me the type definition for this", False),
    ("what is this constant used for", False),

    # ── Vietnamese — deep research ───────────────────────────────────────────
    ("trace luồng xử lý từ đầu đến cuối", True),
    ("ai gọi hàm này", True),
    ("hàm này được gọi từ đâu", True),
    ("trace flow của request này", True),
    ("search kĩ để hiểu rõ luồng hoạt động", True),
    ("search kĩ thật kĩ đi", True),
    ("tìm hiểu sâu về cách hoạt động của module này", True),
    ("tìm hiểu kĩ về cách 2 stage retrieval hoạt động", True),
    ("theo dõi luồng dữ liệu từ API đến database", True),
    ("ảnh hưởng của việc thay đổi file này là gì", True),
    ("nghiên cứu kĩ về cách hệ thống xử lý auth", True),
    ("tìm tất cả các file phụ thuộc vào module này", True),
    ("theo dõi toàn bộ execution path", True),
    ("phân tích sâu dependency graph", True),
    ("tìm hiểu cặn kẽ về pipeline xử lý", True),
    ("map toàn bộ call chain của feature này", True),
    ("tìm tất cả nơi gọi hàm này", True),
    ("blast radius khi thay đổi file này là gì", True),
    ("trace luồng auth từ request đến response", True),
    ("nghiên cứu đầy đủ về event handling chain", True),

    # ── Vietnamese — quick ask ───────────────────────────────────────────────
    ("hàm này làm gì", False),
    ("file này dùng để làm gì", False),
    ("class này có tác dụng gì", False),
    ("tech stack là gì", False),
    ("framework nào được dùng", False),
    ("database schema trông như thế nào", False),
    ("entry point ở đâu", False),
    ("hàm này trả về gì", False),
    ("giải thích module này", False),
    ("biến này dùng để làm gì", False),
    ("tìm file config", False),
    ("API routes được định nghĩa ở đâu", False),
    ("hàm này nhận tham số gì", False),
    ("interface này có nghĩa gì", False),
    ("constant này dùng để làm gì", False),

    # ── Chinese Simplified — deep research ───────────────────────────────────
    ("追踪调用链从认证中间件开始", True),
    ("谁调用了这个函数", True),
    ("这个函数在哪里被调用", True),
    ("追踪请求的完整执行路径", True),
    ("深入研究这个模块的工作原理", True),
    ("仔细搜索整个代码库", True),
    ("分析修改这个文件的影响范围", True),
    ("追踪数据从API到数据库的流程", True),
    ("找出所有依赖这个模块的文件", True),
    ("全面分析依赖关系图", True),
    ("深度追踪认证流程的每个步骤", True),

    # ── Chinese Simplified — quick ask ───────────────────────────────────────
    ("这个函数是做什么的", False),
    ("解释这个类", False),
    ("使用了什么技术栈", False),
    ("数据库结构是什么样的", False),
    ("入口点在哪里", False),
    ("这个方法返回什么", False),
    ("这个框架是什么", False),
    ("找到用户服务类", False),

    # ── Chinese Traditional — deep research ──────────────────────────────────
    ("追蹤呼叫鏈從認證中間件開始", True),
    ("深入研究這個模組的工作原理", True),
    ("分析修改這個檔案的影響範圍", True),
    ("找出所有依賴這個模組的檔案", True),

    # ── Chinese Traditional — quick ask ──────────────────────────────────────
    ("這個函數是做什麼的", False),
    ("解釋這個類別", False),
    ("使用了什麼技術棧", False),

    # ── Japanese — deep research ─────────────────────────────────────────────
    ("この関数の呼び出しチェーンを追跡する", True),
    ("誰がこの関数を呼び出しているか", True),
    ("リクエストの実行パスを追跡する", True),
    ("このモジュールを変更した場合の影響範囲", True),
    ("データフローをエンドツーエンドで追跡する", True),
    ("このファイルに依存するすべてのファイルを見つける", True),
    ("詳細にコードベースを調査する", True),
    ("認証フローを深く掘り下げて調査する", True),

    # ── Japanese — quick ask ─────────────────────────────────────────────────
    ("この関数は何をしますか", False),
    ("このクラスの目的を説明する", False),
    ("使用している技術スタックは何ですか", False),
    ("エントリーポイントはどこですか", False),
    ("このメソッドは何を返しますか", False),
    ("データベーススキーマを見せてください", False),

    # ── Korean — deep research ───────────────────────────────────────────────
    ("이 함수의 호출 체인을 추적하라", True),
    ("누가 이 함수를 호출하는가", True),
    ("요청 실행 경로를 추적하라", True),
    ("이 파일을 변경하면 어떤 영향이 있는가", True),
    ("데이터 흐름을 처음부터 끝까지 추적하라", True),
    ("이 모듈에 의존하는 모든 파일을 찾아라", True),
    ("자세히 코드베이스를 조사하라", True),

    # ── Korean — quick ask ───────────────────────────────────────────────────
    ("이 함수는 무엇을 하는가", False),
    ("이 클래스의 목적을 설명하라", False),
    ("사용 중인 기술 스택은 무엇인가", False),
    ("진입점은 어디인가", False),

    # ── French — deep research ───────────────────────────────────────────────
    ("tracer la chaîne d'appels depuis le middleware", True),
    ("qui appelle cette fonction", True),
    ("tracer le flux d'exécution de la requête", True),
    ("analyser l'impact de modifier ce fichier", True),
    ("trouver tous les fichiers qui dépendent de ce module", True),
    ("rechercher exhaustivement dans le code", True),
    ("analyser en profondeur le graphe de dépendances", True),
    ("suivre le chemin complet des données", True),

    # ── French — quick ask ───────────────────────────────────────────────────
    ("que fait cette fonction", False),
    ("expliquer cette classe", False),
    ("quelle technologie utilise ce projet", False),
    ("où se trouve le point d'entrée", False),
    ("que retourne cette méthode", False),
    ("afficher le schéma de la base de données", False),

    # ── Spanish — deep research ──────────────────────────────────────────────
    ("rastrear la cadena de llamadas desde el middleware", True),
    ("quién llama a esta función", True),
    ("rastrear el flujo de ejecución de la solicitud", True),
    ("analizar el impacto de cambiar este archivo", True),
    ("encontrar todos los archivos que dependen de este módulo", True),
    ("buscar exhaustivamente en el código", True),
    ("analizar en profundidad el grafo de dependencias", True),
    ("seguir el camino completo de los datos", True),

    # ── Spanish — quick ask ──────────────────────────────────────────────────
    ("qué hace esta función", False),
    ("explicar esta clase", False),
    ("qué tecnología usa este proyecto", False),
    ("dónde está el punto de entrada", False),
    ("qué devuelve este método", False),

    # ── German — deep research ───────────────────────────────────────────────
    ("Aufrufkette ab dem Middleware verfolgen", True),
    ("wer ruft diese Funktion auf", True),
    ("den Ausführungspfad der Anfrage verfolgen", True),
    ("Auswirkungen der Änderung dieser Datei analysieren", True),
    ("alle Dateien finden die von diesem Modul abhängen", True),
    ("gründlich den gesamten Code durchsuchen", True),
    ("tief in das Abhängigkeitsgraph eintauchen", True),

    # ── German — quick ask ───────────────────────────────────────────────────
    ("was macht diese Funktion", False),
    ("erkläre diese Klasse", False),
    ("welche Technologie verwendet dieses Projekt", False),
    ("wo ist der Einstiegspunkt", False),

    # ── Portuguese — deep research ───────────────────────────────────────────
    ("rastrear a cadeia de chamadas do middleware", True),
    ("quem chama esta função", True),
    ("rastrear o fluxo de execução da requisição", True),
    ("analisar o impacto de modificar este arquivo", True),
    ("encontrar todos os arquivos que dependem deste módulo", True),
    ("pesquisar exaustivamente no código", True),
    ("mergulhar fundo no grafo de dependências", True),

    # ── Portuguese — quick ask ───────────────────────────────────────────────
    ("o que esta função faz", False),
    ("explicar esta classe", False),
    ("qual tecnologia este projeto usa", False),
    ("onde fica o ponto de entrada", False),

    # ── Italian — deep research ──────────────────────────────────────────────
    ("tracciare la catena di chiamate dal middleware", True),
    ("chi chiama questa funzione", True),
    ("tracciare il flusso di esecuzione della richiesta", True),
    ("analizzare l'impatto di modificare questo file", True),
    ("trovare tutti i file che dipendono da questo modulo", True),
    ("ricercare esaustivamente nel codice", True),
    ("immergersi in profondità nel grafo delle dipendenze", True),

    # ── Italian — quick ask ──────────────────────────────────────────────────
    ("cosa fa questa funzione", False),
    ("spiegare questa classe", False),
    ("quale tecnologia usa questo progetto", False),
    ("dove si trova il punto di ingresso", False),

    # ── Russian — deep research ──────────────────────────────────────────────
    ("отследить цепочку вызовов от middleware", True),
    ("кто вызывает эту функцию", True),
    ("отследить путь выполнения запроса", True),
    ("проанализировать влияние изменения этого файла", True),
    ("найти все файлы зависящие от этого модуля", True),
    ("тщательно найти в коде", True),
    ("глубоко изучить граф зависимостей", True),
    ("проследить поток данных сквозь систему", True),

    # ── Russian — quick ask ──────────────────────────────────────────────────
    ("что делает эта функция", False),
    ("объяснить этот класс", False),
    ("какие технологии используются", False),
    ("где находится точка входа", False),

    # ── Arabic — deep research ───────────────────────────────────────────────
    ("تتبع سلسلة الاستدعاء من البرمجيات الوسيطة", True),
    ("من يستدعي هذه الدالة", True),
    ("تتبع مسار تنفيذ الطلب", True),
    ("تحليل تأثير تعديل هذا الملف", True),
    ("البحث بعمق في الكود", True),
    ("تتبع تدفق البيانات من البداية إلى النهاية", True),

    # ── Arabic — quick ask ───────────────────────────────────────────────────
    ("ماذا تفعل هذه الدالة", False),
    ("اشرح هذا الفصل", False),
    ("ما هي التقنية المستخدمة", False),
    ("أين نقطة الدخول", False),

    # ── Hindi — deep research ────────────────────────────────────────────────
    ("middleware से कॉल चेन को ट्रेस करें", True),
    ("इस फ़ंक्शन को कौन कॉल करता है", True),
    ("रिक्वेस्ट का पूरा execution path ट्रेस करें", True),
    ("इस फ़ाइल को बदलने का प्रभाव विश्लेषण करें", True),
    ("इस मॉड्यूल पर निर्भर सभी फ़ाइलें खोजें", True),
    ("कोड में गहराई से खोजें", True),

    # ── Hindi — quick ask ────────────────────────────────────────────────────
    ("यह फ़ंक्शन क्या करता है", False),
    ("इस क्लास को समझाएं", False),
    ("कौन सी तकनीक उपयोग की गई है", False),
    ("entry point कहाँ है", False),

    # ── Thai — deep research ─────────────────────────────────────────────────
    ("ติดตามสายการเรียกจาก middleware", True),
    ("ใครเรียกฟังก์ชันนี้", True),
    ("ติดตามเส้นทางการทำงานของ request", True),
    ("วิเคราะห์ผลกระทบของการเปลี่ยนแปลงไฟล์นี้", True),
    ("ค้นหาไฟล์ทั้งหมดที่ขึ้นอยู่กับโมดูลนี้", True),
    ("ค้นหาอย่างละเอียดถี่ถ้วน", True),

    # ── Thai — quick ask ─────────────────────────────────────────────────────
    ("ฟังก์ชันนี้ทำอะไร", False),
    ("อธิบายคลาสนี้", False),
    ("ใช้เทคโนโลยีอะไร", False),
    ("จุดเริ่มต้นอยู่ที่ไหน", False),

    # ── Indonesian — deep research ───────────────────────────────────────────
    ("lacak rantai panggilan dari middleware", True),
    ("siapa yang memanggil fungsi ini", True),
    ("lacak jalur eksekusi permintaan ini", True),
    ("analisis dampak mengubah file ini", True),
    ("temukan semua file yang bergantung pada modul ini", True),
    ("cari secara menyeluruh di seluruh kode", True),
    ("ikuti aliran data dari API ke database", True),

    # ── Indonesian — quick ask ───────────────────────────────────────────────
    ("apa yang dilakukan fungsi ini", False),
    ("jelaskan kelas ini", False),
    ("teknologi apa yang digunakan", False),
    ("di mana titik masuknya", False),

    # ── Malay — deep research ────────────────────────────────────────────────
    ("kesan perubahan pada fail ini", True),
    ("siapa yang memanggil fungsi ini dalam keseluruhan sistem", True),
    ("jejaki aliran permintaan dari mula hingga akhir", True),
    ("cari semua fail yang bergantung kepada modul ini", True),

    # ── Malay — quick ask ────────────────────────────────────────────────────
    ("apakah fungsi ini lakukan", False),
    ("jelaskan kelas ini", False),
    ("teknologi apa yang digunakan", False),

    # ── Dutch — deep research ────────────────────────────────────────────────
    ("de aanroepketen van de middleware traceren", True),
    ("wie roept deze functie aan", True),
    ("het uitvoeringspad van het verzoek traceren", True),
    ("de impact analyseren van het wijzigen van dit bestand", True),
    ("alle bestanden vinden die afhankelijk zijn van deze module", True),
    ("grondig zoeken door de codebase", True),

    # ── Dutch — quick ask ────────────────────────────────────────────────────
    ("wat doet deze functie", False),
    ("leg deze klasse uit", False),
    ("welke technologie wordt gebruikt", False),
    ("waar is het toegangspunt", False),

    # ── Polish — deep research ───────────────────────────────────────────────
    ("śledzić łańcuch wywołań od middleware", True),
    ("kto wywołuje tę funkcję", True),
    ("śledzić ścieżkę wykonania żądania", True),
    ("przeanalizować wpływ zmiany tego pliku", True),
    ("znaleźć wszystkie pliki zależne od tego modułu", True),
    ("przeszukać dokładnie cały kod", True),

    # ── Polish — quick ask ───────────────────────────────────────────────────
    ("co robi ta funkcja", False),
    ("wyjaśnij tę klasę", False),
    ("jakiej technologii używa ten projekt", False),
    ("gdzie jest punkt wejścia", False),

    # ── Turkish — deep research ──────────────────────────────────────────────
    ("middleware'den başlayan çağrı zincirini takip et", True),
    ("bu fonksiyonu kim çağırıyor", True),
    ("isteğin yürütme yolunu takip et", True),
    ("bu dosyayı değiştirmenin etkisini analiz et", True),
    ("bu modüle bağlı tüm dosyaları bul", True),
    ("kod tabanını kapsamlı şekilde ara", True),

    # ── Turkish — quick ask ──────────────────────────────────────────────────
    ("bu fonksiyon ne yapıyor", False),
    ("bu sınıfı açıkla", False),
    ("bu proje hangi teknolojiyi kullanıyor", False),
    ("giriş noktası nerede", False),

    # ── Swedish — deep research ──────────────────────────────────────────────
    ("spåra anropskedjan från middleware", True),
    ("vem anropar denna funktion", True),
    ("spåra exekveringsvägen för begäran", True),
    ("analysera effekten av att ändra den här filen", True),
    ("hitta alla filer som beror på den här modulen", True),
    ("sök noggrant igenom hela kodbasen", True),

    # ── Swedish — quick ask ──────────────────────────────────────────────────
    ("vad gör den här funktionen", False),
    ("förklara den här klassen", False),
    ("vilken teknik använder det här projektet", False),

    # ── Finnish — deep research ──────────────────────────────────────────────
    ("jäljitä kutsuketju middleware:stä", True),
    ("kuka kutsuu tätä funktiota", True),
    ("jäljitä pyynnön suorituspolku", True),
    ("analysoi tämän tiedoston muuttamisen vaikutus", True),
    ("etsi kaikki tähän moduuliin riippuvat tiedostot", True),

    # ── Finnish — quick ask ──────────────────────────────────────────────────
    ("mitä tämä funktio tekee", False),
    ("selitä tämä luokka", False),
    ("mitä teknologiaa tämä projekti käyttää", False),

    # ── Norwegian — deep research ────────────────────────────────────────────
    ("spore anropskjeden fra mellomvaren", True),
    ("hvem kaller denne funksjonen", True),
    ("spore utførelsesveien for forespørselen", True),
    ("analysere effekten av å endre denne filen", True),
    ("finn alle filer som er avhengige av denne modulen", True),

    # ── Norwegian — quick ask ────────────────────────────────────────────────
    ("hva gjør denne funksjonen", False),
    ("forklar denne klassen", False),
    ("hvilken teknologi bruker dette prosjektet", False),

    # ── Hungarian — deep research ────────────────────────────────────────────
    ("a hívási lánc nyomon követése a middleware-től", True),
    ("ki hívja ezt a funkciót", True),
    ("a kérés végrehajtási útvonalának nyomon követése", True),
    ("ennek a fájlnak a módosítási hatásának elemzése", True),
    ("az összes erre a modulra támaszkodó fájl megkeresése", True),

    # ── Hungarian — quick ask ────────────────────────────────────────────────
    ("mit csinál ez a funkció", False),
    ("magyarázd el ezt az osztályt", False),
    ("milyen technológiát használ ez a projekt", False),

    # ── Romanian — deep research ─────────────────────────────────────────────
    ("urmăriți lanțul de apeluri de la middleware", True),
    ("cine apelează această funcție", True),
    ("urmăriți calea de execuție a cererii", True),
    ("analizați impactul modificării acestui fișier", True),
    ("găsiți toate fișierele care depind de acest modul", True),

    # ── Romanian — quick ask ─────────────────────────────────────────────────
    ("ce face această funcție", False),
    ("explicați această clasă", False),
    ("ce tehnologie folosește acest proiect", False),

    # ── Ukrainian — deep research ────────────────────────────────────────────
    ("відстежити ланцюжок викликів від middleware", True),
    ("хто викликає цю функцію", True),
    ("відстежити шлях виконання запиту", True),
    ("проаналізувати вплив зміни цього файлу", True),
    ("знайти всі файли які залежать від цього модуля", True),

    # ── Ukrainian — quick ask ────────────────────────────────────────────────
    ("що робить ця функція", False),
    ("поясни цей клас", False),
    ("яку технологію використовує цей проект", False),

    # ── Persian — deep research ──────────────────────────────────────────────
    ("زنجیره فراخوانی را از middleware ردیابی کنید", True),
    ("چه کسی این تابع را فراخوانی می‌کند", True),
    ("مسیر اجرای درخواست را دنبال کنید", True),
    ("تأثیر تغییر این فایل را تحلیل کنید", True),
    ("همه فایل‌هایی که به این ماژول وابسته‌اند را پیدا کنید", True),

    # ── Persian — quick ask ──────────────────────────────────────────────────
    ("این تابع چه کاری انجام می‌دهد", False),
    ("این کلاس را توضیح دهید", False),
    ("این پروژه از چه فناوری‌ای استفاده می‌کند", False),

    # ── Hebrew — deep research ───────────────────────────────────────────────
    ("עקוב אחרי שרשרת הקריאות מה-middleware", True),
    ("מי קורא לפונקציה זו", True),
    ("עקוב אחרי נתיב הביצוע של הבקשה", True),
    ("נתח את ההשפעה של שינוי קובץ זה", True),
    ("מצא את כל הקבצים שתלויים במודול זה", True),

    # ── Hebrew — quick ask ───────────────────────────────────────────────────
    ("מה עושה הפונקציה הזו", False),
    ("הסבר את המחלקה הזו", False),
    ("איזו טכנולוגיה הפרויקט הזה משתמש", False),

    # ── Czech — deep research ────────────────────────────────────────────────
    ("sledovat řetězec volání od middleware", True),
    ("kdo volá tuto funkci", True),
    ("sledovat cestu provádění požadavku", True),
    ("analyzovat dopad změny tohoto souboru", True),
    ("najít všechny soubory závislé na tomto modulu", True),

    # ── Czech — quick ask ────────────────────────────────────────────────────
    ("co dělá tato funkce", False),
    ("vysvětlit tuto třídu", False),
    ("jakou technologii tento projekt používá", False),

    # ── Danish — deep research ───────────────────────────────────────────────
    ("spor opkaldskæden fra middlewaren", True),
    ("hvem kalder denne funktion", True),
    ("spor udførelsesstien for anmodningen", True),
    ("analysere effekten af at ændre denne fil", True),
    ("find alle filer der afhænger af dette modul", True),

    # ── Danish — quick ask ───────────────────────────────────────────────────
    ("hvad gør denne funktion", False),
    ("forklar denne klasse", False),
    ("hvilken teknologi bruger dette projekt", False),

    # ── Greek — deep research ────────────────────────────────────────────────
    ("παρακολούθηση της αλυσίδας κλήσεων από το middleware", True),
    ("ποιος καλεί αυτή τη συνάρτηση", True),
    ("παρακολούθηση της διαδρομής εκτέλεσης αιτήματος", True),
    ("ανάλυση της επίδρασης τροποποίησης αυτού του αρχείου", True),

    # ── Greek — quick ask ────────────────────────────────────────────────────
    ("τι κάνει αυτή η συνάρτηση", False),
    ("εξηγήστε αυτή την κλάση", False),
    ("ποια τεχνολογία χρησιμοποιεί αυτό το έργο", False),

    # ── Bengali — deep research ──────────────────────────────────────────────
    ("middleware থেকে কল চেইন ট্রেস করুন", True),
    ("কে এই ফাংশন কল করে", True),
    ("রিকোয়েস্টের এক্সিকিউশন পাথ ট্রেস করুন", True),
    ("এই ফাইল পরিবর্তনের প্রভাব বিশ্লেষণ করুন", True),

    # ── Bengali — quick ask ──────────────────────────────────────────────────
    ("এই ফাংশন কী করে", False),
    ("এই ক্লাস ব্যাখ্যা করুন", False),
    ("কী প্রযুক্তি ব্যবহার করা হয়েছে", False),

    # ── Mixed / code-switching (common in dev communities) ───────────────────
    ("trace luồng của auth middleware call", True),
    ("ai call hàm này trong toàn bộ system", True),
    ("search kĩ về flow của request này", True),
    ("hàm này được call từ đâu trong codebase", True),
    ("ảnh hưởng nếu tôi thay đổi file này là gì", True),
    ("tìm hết dependency của module này", True),
    ("deep dive vào authentication flow", True),
    ("研究 auth middleware 的完整调用链", True),
    ("追踪 request 从 API 到 database 的流程", True),
    ("who calls この関数", True),
    ("이 function의 call chain을 trace해줘", True),
    ("trace the luồng từ controller đến service đến repo", True),

    ("hàm này làm gì vậy", False),
    ("file này là file gì", False),
    ("tech stack của project này là gì", False),
    ("hàm này return gì", False),
    ("config này có nghĩa gì", False),
    ("このファイルは何のためにありますか", False),
    ("이 클래스가 뭐하는 거야", False),
    ("此函数做什么", False),
]


# ---------------------------------------------------------------------------
# Character n-gram extraction
# ---------------------------------------------------------------------------

def _char_ngrams(text: str, n_min: int = 2, n_max: int = 4) -> list[str]:
    """Extract character n-grams from lowercased text (language-agnostic)."""
    text = " " + text.lower().strip() + " "
    result: list[str] = []
    for n in range(n_min, n_max + 1):
        for i in range(len(text) - n + 1):
            result.append(text[i : i + n])
    return result


# ---------------------------------------------------------------------------
# Primary: sklearn TF-IDF (char_wb n-gram) + Logistic Regression
# ---------------------------------------------------------------------------

def _build_sklearn_classifier(dataset: list[tuple[str, bool]]) -> "object | None":
    """Build a sklearn classifier. Returns None if sklearn is not available."""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore[import]
        from sklearn.linear_model import LogisticRegression           # type: ignore[import]
        import numpy as np                                             # type: ignore[import]

        texts = [t for t, _ in dataset]
        labels = np.array([1 if d else 0 for _, d in dataset])

        vec = TfidfVectorizer(
            analyzer="char_wb",     # word-boundary padded char n-grams
            ngram_range=(2, 4),     # bigrams through 4-grams
            sublinear_tf=True,      # log(1 + tf) scaling
            min_df=1,
            strip_accents=None,     # preserve all Unicode characters
        )
        X = vec.fit_transform(texts)
        clf = LogisticRegression(max_iter=500, C=5.0, solver="lbfgs")
        clf.fit(X, labels)

        class _SklearnClassifier:
            def predict(self, text: str, threshold: float = 0.5) -> tuple[bool, float]:
                xv = vec.transform([text])
                prob = clf.predict_proba(xv)[0, 1]
                return prob >= threshold, float(prob)

        return _SklearnClassifier()

    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Fallback: pure Python log-odds Naive Bayes (no dependencies)
# ---------------------------------------------------------------------------

class _NaiveBayesClassifier:
    """
    Log-odds Naive Bayes classifier trained on char n-gram TF features.
    Pure Python fallback when sklearn is unavailable.
    """

    def __init__(self, dataset: list[tuple[str, bool]]) -> None:
        deep_count: Counter[str] = Counter()
        quick_count: Counter[str] = Counter()
        deep_doc_count = 0
        quick_doc_count = 0

        for text, is_deep in dataset:
            ngrams = _char_ngrams(text)
            if is_deep:
                deep_count.update(ngrams)
                deep_doc_count += 1
            else:
                quick_count.update(ngrams)
                quick_doc_count += 1

        deep_total = sum(deep_count.values()) or 1
        quick_total = sum(quick_count.values()) or 1
        all_ngrams = set(deep_count) | set(quick_count)
        vocab_size = len(all_ngrams)
        smoothing = 1.0

        self._weights: dict[str, float] = {}
        for ng in all_ngrams:
            p_deep = (deep_count[ng] + smoothing) / (deep_total + smoothing * vocab_size)
            p_quick = (quick_count[ng] + smoothing) / (quick_total + smoothing * vocab_size)
            self._weights[ng] = math.log(p_deep / p_quick)

        self._prior = math.log(deep_doc_count / max(quick_doc_count, 1))

    def _score(self, text: str) -> float:
        ngrams = _char_ngrams(text)
        return self._prior + sum(self._weights.get(ng, 0.0) for ng in ngrams)

    def predict(self, text: str, threshold: float = 0.5) -> tuple[bool, float]:
        """Return (is_deep_research, score). threshold is in log-odds space."""
        s = self._score(text)
        # Convert log-odds to probability for a consistent return type
        prob = 1.0 / (1.0 + math.exp(-s))
        return prob >= threshold, prob


# ---------------------------------------------------------------------------
# Disk persistence helpers
# ---------------------------------------------------------------------------

def _data_dir() -> Path:
    return Path(os.getenv("CODESPECTRA_DATA_DIR", "."))


def _model_path() -> Path:
    return _data_dir() / "classifier_model.pkl"


def _meta_path() -> Path:
    return _data_dir() / "classifier_meta.json"


def _save(clf: object, backend: str, total_examples: int) -> None:
    """Serialise the trained model + metadata to disk."""
    _data_dir().mkdir(parents=True, exist_ok=True)
    try:
        try:
            import joblib  # type: ignore[import]
            joblib.dump(clf, _model_path())
        except ImportError:
            with open(_model_path(), "wb") as f:
                pickle.dump(clf, f)
        _meta_path().write_text(json.dumps({
            "backend": backend,
            "total_examples": total_examples,
            "trained_at": datetime.now(timezone.utc).isoformat(),
        }))
    except Exception as exc:
        import logging
        logging.getLogger("codespectra").warning("classifier: save failed: %s", exc)


def _load() -> "tuple[object, str] | None":
    """Load the persisted model from disk. Returns (clf, backend) or None."""
    if not _model_path().exists():
        return None
    try:
        try:
            import joblib  # type: ignore[import]
            clf = joblib.load(_model_path())
        except (ImportError, Exception):
            with open(_model_path(), "rb") as f:
                clf = pickle.load(f)
        meta: dict = {}
        if _meta_path().exists():
            meta = json.loads(_meta_path().read_text())
        backend = meta.get("backend", "unknown")
        return clf, backend
    except Exception as exc:
        import logging
        logging.getLogger("codespectra").warning("classifier: load failed, will retrain: %s", exc)
        return None


# ---------------------------------------------------------------------------
# In-memory singleton
# ---------------------------------------------------------------------------

_classifier: "object | None" = None
_backend: str = "not_trained"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify_intent(text: str, threshold: float = 0.5) -> tuple[bool, float]:
    """Return (is_deep_research, confidence_probability).

    On first call: tries to load persisted model from disk; if absent, trains fresh
    and saves. Subsequent calls use the in-memory singleton — <1 ms per prediction.
    """
    global _classifier, _backend  # noqa: PLW0603
    if _classifier is None:
        loaded = _load()
        if loaded is not None:
            _classifier, _backend = loaded
        else:
            _classifier, _backend = _train_and_save(list(_DATASET))
    return _classifier.predict(text, threshold=threshold)  # type: ignore[union-attr]


def warm_up() -> None:
    """Force classifier initialisation. Call once at startup in a background task."""
    classify_intent("warm up call")


def retrain_with_examples(extra: list[tuple[str, bool]]) -> None:
    """Rebuild the classifier with built-in dataset + user-added examples, then persist."""
    global _classifier, _backend  # noqa: PLW0603
    combined = list(_DATASET) + extra
    _classifier, _backend = _train_and_save(combined)


def get_status() -> dict:
    """Return classifier status, preferring in-memory state over disk for accuracy."""
    if _classifier is not None:
        return {
            "trained": True,
            "backend": _backend,
            "builtin_examples": len(_DATASET),
        }
    model_exists = _model_path().exists()
    meta: dict = {}
    if _meta_path().exists():
        try:
            meta = json.loads(_meta_path().read_text())
        except Exception:
            pass
    return {
        "trained": model_exists,
        "backend": meta.get("backend", "not_trained") if model_exists else "not_trained",
        "builtin_examples": len(_DATASET),
    }


# ---------------------------------------------------------------------------
# Internal: train + save
# ---------------------------------------------------------------------------

def _train_and_save(dataset: list[tuple[str, bool]]) -> "tuple[object, str]":
    sklearn_clf = _build_sklearn_classifier(dataset)
    if sklearn_clf is not None:
        backend = "sklearn"
        clf = sklearn_clf
    else:
        backend = "pure_python"
        clf = _NaiveBayesClassifier(dataset)
    _save(clf, backend, len(dataset))
    return clf, backend
