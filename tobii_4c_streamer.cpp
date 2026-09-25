//
// Βήματα πριν το build:
//   1. Κατέβασε το "Stream Engine" (consumer) SDK από:
//      https://developer.tobii.com/  ->  Consumer eye trackers -> Stream Engine
//      (ή μέσω NuGet: Tobii.StreamEngine.Native)
//   2. Θα πάρεις: include/tobii/tobii.h, include/tobii/tobii_streams.h,
//      lib/x64/tobii_stream_engine.lib, lib/x64/tobii_stream_engine.dll
//   3. Βάλε τα σε ένα φάκελο "tobii_sdk/" δίπλα σε αυτό το .cpp
//      (δες CMakeLists.txt).
//
//
//   mkdir -p build
//   cd build
//   cmake .. -G "MinGW Makefiles"
//   cmake --build .
//   cd ..
//   cp build/tobii_4c_streamer.exe .
//
//
// 1. Δημιουργία build directory και build με CMake
//  mkdir -p build
//  cd build
//  cmake .. -A x64
//  cmake --build . --config Release
//  cd ..
//
//  2. Έλεγχος ότι το exe και το dll βρίσκονται μαζί (το CMakeLists.txt το κάνει αυτόματα, αλλά καλό να επιβεβαιώσεις)
//  ls build/Release/tobii_4c_streamer.exe
//  ls build/Release/tobii_stream_engine.dll
//
//  3. Αντιγραφή δίπλα στο main.py
//  cp build/Release/tobii_4c_streamer.exe .
//  cp build/Release/tobii_stream_engine.dll .

//  4. Αυτόνομη δοκιμή — πρέπει να δεις γραμμές x,y,valid,timestamp να τρέχουν όσο κοιτάς την οθόνη (Ctrl+C για έξοδο)
//  ./tobii_4c_streamer.exe
//
#include "tobii/tobii.h"
#include "tobii/tobii_streams.h"

#include <cstdio>
#include <cstring>
#include <atomic>
#include <csignal>

namespace {

std::atomic<bool> g_running{true};

void on_sigint(int) { g_running = false; }

// Μαζεύει το πρώτο URL συσκευής που βρίσκεται.
void url_receiver(char const* url, void* user_data) {
    char* buffer = static_cast<char*>(user_data);
    if (*buffer != '\0') return;  // κράτα μόνο το πρώτο
    if (std::strlen(url) < 256) {
        std::strcpy(buffer, url);
    }
}

// Καλείται από το SDK σε κάθε νέο δείγμα βλέμματος.
void gaze_point_callback(tobii_gaze_point_t const* gaze_point, void* /*user_data*/) {
        std::fprintf(
        stdout, "%.6f,%.6f,%d,%lld\n",
        gaze_point->position_xy[0],
        gaze_point->position_xy[1],
        gaze_point->validity == TOBII_VALIDITY_VALID ? 1 : 0,
        static_cast<long long>(0)  // βλ. ΣΗΜΕΙΩΣΗ timestamp παρακάτω
    );
    std::fflush(stdout);
}

}  // namespace

int main() {
    std::signal(SIGINT, on_sigint);

    tobii_api_t* api = nullptr;
    tobii_error_t result = tobii_api_create(&api, nullptr, nullptr);
    if (result != TOBII_ERROR_NO_ERROR) {
        std::fprintf(stderr, "ERROR tobii_api_create: %s\n", tobii_error_message(result));
        return 1;
    }

    char url[256] = {0};
    result = tobii_enumerate_local_device_urls(api, url_receiver, url);
    if (result != TOBII_ERROR_NO_ERROR || url[0] == '\0') {
        std::fprintf(stderr, "ERROR: Δεν βρέθηκε συνδεδεμένη συσκευή Tobii.\n");
        tobii_api_destroy(api);
        return 1;
    }
    std::fprintf(stderr, "INFO: Βρέθηκε συσκευή: %s\n", url);

    tobii_device_t* device = nullptr;
        result = tobii_device_create(api, url, TOBII_FIELD_OF_USE_INTERACTIVE, &device);
    if (result != TOBII_ERROR_NO_ERROR) {
        std::fprintf(stderr, "ERROR tobii_device_create: %s\n", tobii_error_message(result));
        tobii_api_destroy(api);
        return 1;
    }

    result = tobii_gaze_point_subscribe(device, gaze_point_callback, nullptr);
    if (result != TOBII_ERROR_NO_ERROR) {
        std::fprintf(stderr, "ERROR tobii_gaze_point_subscribe: %s\n", tobii_error_message(result));
        tobii_device_destroy(device);
        tobii_api_destroy(api);
        return 1;
    }

    std::fprintf(stderr, "INFO: Subscribed. Streaming gaze data (Ctrl+C για έξοδο)...\n");

    while (g_running) {
        // Μπλοκάρει έως 100ms περιμένοντας δεδομένα, μετά επεξεργάζεται ό,τι έχει έρθει (καλεί το gaze_point_callback).
        result = tobii_wait_for_callbacks(1, &device);
        if (result != TOBII_ERROR_NO_ERROR && result != TOBII_ERROR_TIMED_OUT) {
            std::fprintf(stderr, "WARN tobii_wait_for_callbacks: %s\n", tobii_error_message(result));
            break;
        }
        result = tobii_device_process_callbacks(device);
        if (result != TOBII_ERROR_NO_ERROR) {
            std::fprintf(stderr, "WARN tobii_device_process_callbacks: %s\n", tobii_error_message(result));
            break;
        }
    }

    tobii_gaze_point_unsubscribe(device);
    tobii_device_destroy(device);
    tobii_api_destroy(api);
    return 0;
}


