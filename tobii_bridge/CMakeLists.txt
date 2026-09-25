cmake_minimum_required(VERSION 3.15)
project(tobii_gaze_bridge CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------------------------------------------------------------------------
# ΣΗΜΑΝΤΙΚΟ: άλλαξε αυτή τη διαδρομή ώστε να δείχνει στον φάκελο του Tobii
# Stream Engine SDK (Tobii.StreamEngine.Native), όπου υπάρχουν:
#   include/tobii/tobii.h
#   include/tobii/tobii_streams.h
#   lib/x64/tobii_stream_engine.lib
#   lib/x64/tobii_stream_engine.dll   (runtime — αντιγράφεται αυτόματα παρακάτω)
#
# Αν το tobii_bridge_CMakeLists.txt (μετονομασμένο σε CMakeLists.txt) βρίσκεται
# ΜΕΣΑ στον φάκελο tobii_bridge/, και το SDK είναι ένα επίπεδο πάνω, σε
# Tobii.StreamEngine.Native.2.2.2.363/, το default παρακάτω πιθανόν δουλεύει
# ήδη χωρίς αλλαγή. Αν όχι, βάλε το απόλυτο path.
# ---------------------------------------------------------------------------
set(TOBII_SDK_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../Tobii.StreamEngine.Native.2.2.2.363"
    CACHE PATH "Path to Tobii Stream Engine SDK root")

if(NOT EXISTS "${TOBII_SDK_DIR}/include/tobii/tobii.h" AND
   NOT EXISTS "${TOBII_SDK_DIR}/build/native/include/tobii/tobii.h")
    message(FATAL_ERROR
        "Δεν βρέθηκε το tobii.h ούτε σε ${TOBII_SDK_DIR}/include/tobii/ "
        "ούτε σε ${TOBII_SDK_DIR}/build/native/include/tobii/ — "
        "όρισε το σωστό TOBII_SDK_DIR:\n"
        "  cmake .. -DTOBII_SDK_DIR=\"C:/πλήρες/path/προς/Tobii.StreamEngine.Native.2.2.2.363\""
    )
endif()

add_executable(tobii_gaze_bridge tobii_gaze_bridge.cpp)

target_include_directories(tobii_gaze_bridge PRIVATE
    "${TOBII_SDK_DIR}/include"
    "${TOBII_SDK_DIR}/build/native/include"
)

# Το ακριβές sub-path της .lib μέσα στο SDK διαφέρει ανά έκδοση/πακέτο —
# ψάχνουμε τις πιο συνηθισμένες τοποθεσίες αυτόματα (επιβεβαιωμένο path
# για το NuGet πακέτο Tobii.StreamEngine.Native 2.2.2.363:
# build/native/lib/x64/).
find_library(TOBII_STREAM_ENGINE_LIB
    NAMES tobii_stream_engine
    PATHS
        "${TOBII_SDK_DIR}/build/native/lib/x64"
        "${TOBII_SDK_DIR}/lib/x64"
        "${TOBII_SDK_DIR}/lib"
        "${TOBII_SDK_DIR}/64/lib"
        "${TOBII_SDK_DIR}/runtimes/win-x64/native"
    NO_DEFAULT_PATH
)
if(NOT TOBII_STREAM_ENGINE_LIB)
    message(FATAL_ERROR
        "Δεν βρέθηκε το tobii_stream_engine.lib μέσα στο ${TOBII_SDK_DIR}. "
        "Έλεγξε τη δομή φακέλων του SDK και πρόσθεσε το σωστό PATHS εδώ."
    )
endif()
target_link_libraries(tobii_gaze_bridge PRIVATE "${TOBII_STREAM_ENGINE_LIB}")

# Αντιγραφή του .dll δίπλα στο .exe μετά το build — χωρίς αυτό, το exe θα
# ξεκινάει αλλά θα κρασάρει αμέσως (missing DLL) όταν τρέχει έξω από αυτόν
# τον φάκελο.
find_file(TOBII_STREAM_ENGINE_DLL
    NAMES tobii_stream_engine.dll
    PATHS
        "${TOBII_SDK_DIR}/build/native/lib/x64"
        "${TOBII_SDK_DIR}/lib/x64"
        "${TOBII_SDK_DIR}/bin/x64"
        "${TOBII_SDK_DIR}/64/lib"
        "${TOBII_SDK_DIR}/runtimes/win-x64/native"
    NO_DEFAULT_PATH
)
if(TOBII_STREAM_ENGINE_DLL)
    add_custom_command(TARGET tobii_gaze_bridge POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${TOBII_STREAM_ENGINE_DLL}"
            "$<TARGET_FILE_DIR:tobii_gaze_bridge>"
        COMMENT "Αντιγραφή tobii_stream_engine.dll δίπλα στο tobii_gaze_bridge.exe"
    )
else()
    message(WARNING
        "Δεν βρέθηκε το tobii_stream_engine.dll αυτόματα — αντέγραψέ το "
        "χειροκίνητα δίπλα στο tobii_gaze_bridge.exe μετά το build."
    )
endif()
