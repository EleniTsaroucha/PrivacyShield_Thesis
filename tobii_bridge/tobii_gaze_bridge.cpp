// tobii_gaze_bridge.cpp
// =====================
// Standalone εφαρμογή κονσόλας που μιλάει απευθείας με τον Tobii Eye
// Tracker 4C μέσω του Tobii STREAM ENGINE SDK.
//
// ΣΗΜΕΙΩΣΗ (διόρθωση): η έκδοση SDK που χρησιμοποιείται εδώ
// (Tobii.StreamEngine.Native 2.2.2.363, headers με ημερομηνία 2018) είναι
// προγενέστερη της εισαγωγής του μηχανισμού "Pro Upgrade Key" licensing
// στο tobii_research — δεν έχει καν το concept TOBII_FIELD_OF_USE_*, το
// tobii_device_create() εδώ παίρνει απλά (api, url, &device), χωρίς
// παράμετρο field-of-use. Οπότε δεν χρειάζεται καθόλου το παλιότερο
// workaround που υπέθετε η αρχική εκδοχή αυτού του αρχείου.
//
// Επικοινωνεί με το main.py μέσω ενός Python subprocess wrapper
// (tobii4c_gaze_tracker.py) χρησιμοποιώντας ένα απλό, γραμμοκεντρικό
// πρωτόκολλο πάνω από stdout:
//
//   READY\t<model>\t<serial>\n          -> μία φορά, μόλις συνδεθεί η συσκευή
//   GAZE\t<timestamp_us>\t<x>\t<y>\t<valid 0|1>\n  -> για κάθε δείγμα βλέμματος
//   ERROR\t<μήνυμα>\n                   -> σε περίπτωση αποτυχίας, μετά exit(1)
//
// Οι συντεταγμένες x, y είναι normalized (0.0–1.0) ως προς την ενεργή
// οθόνη, ίδια σύμβαση με το tobii_research (Pro SDK) gaze_point.
//
// Χρήση:
//   tobii_gaze_bridge.exe                       (πρώτη διαθέσιμη συσκευή)
//   tobii_gaze_bridge.exe --serial ABC123        (φιλτράρισμα με serial)
//   tobii_gaze_bridge.exe --model-contains 4C    (φιλτράρισμα με substring μοντέλου)

#include <tobii/tobii.h>
#include <tobii/tobii_streams.h>

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>
#include <atomic>
#include <csignal>

namespace {

std::atomic<bool> g_running{true};

void on_sigint(int) {
    g_running.store(false);
}

// --- URL enumeration -------------------------------------------------
struct UrlList {
    std::vector<std::string> urls;
};

void url_receiver(char const* url, void* user_data) {
    auto* list = static_cast<UrlList*>(user_data);
    list->urls.emplace_back(url);
}

// --- Gaze point callback ----------------------------------------------
void gaze_point_callback(tobii_gaze_point_t const* gaze_point, void* /*user_data*/) {
    // validity: TOBII_VALIDITY_VALID (1) ή TOBII_VALIDITY_INVALID (0)
    int valid = (gaze_point->validity == TOBII_VALIDITY_VALID) ? 1 : 0;
    std::printf(
        "GAZE\t%lld\t%.6f\t%.6f\t%d\n",
        static_cast<long long>(gaze_point->timestamp_us),
        gaze_point->position_xy[0],
        gaze_point->position_xy[1],
        valid
    );
    std::fflush(stdout);
}

// --- Command-line parsing ----------------------------------------------
struct Args {
    std::string serial;         // --serial
    std::string model_contains; // --model-contains
};

Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--serial" && i + 1 < argc) {
            a.serial = argv[++i];
        } else if (arg == "--model-contains" && i + 1 < argc) {
            a.model_contains = argv[++i];
        }
    }
    return a;
}

void report_error(const std::string& msg) {
    std::printf("ERROR\t%s\n", msg.c_str());
    std::fflush(stdout);
}

} // namespace

int main(int argc, char** argv) {
    std::signal(SIGINT, on_sigint);
    std::signal(SIGTERM, on_sigint);

    Args args = parse_args(argc, argv);

    // --- 1. Δημιουργία API -------------------------------------------
    tobii_api_t* api = nullptr;
    tobii_error_t result = tobii_api_create(&api, nullptr, nullptr);
    if (result != TOBII_ERROR_NO_ERROR) {
        report_error("tobii_api_create απέτυχε");
        return 1;
    }

    // --- 2. Enumerate συνδεδεμένες συσκευές ---------------------------
    UrlList list;
    result = tobii_enumerate_local_device_urls(api, url_receiver, &list);
    if (result != TOBII_ERROR_NO_ERROR || list.urls.empty()) {
        report_error("Δεν βρέθηκε καμία συνδεδεμένη συσκευή Tobii (enumerate)");
        tobii_api_destroy(api);
        return 1;
    }

    // --- 3. Επιλογή συσκευής (με βάση serial/model αν δόθηκαν) --------
    tobii_device_t* device = nullptr;
    std::string chosen_url;
    std::string chosen_model, chosen_serial;

    for (const auto& url : list.urls) {
        tobii_device_t* candidate = nullptr;
        // Αυτή η έκδοση SDK (2018) δεν έχει field-of-use παράμετρο —
        // tobii_device_create παίρνει μόνο (api, url, &device).
        result = tobii_device_create(api, url.c_str(), &candidate);
        if (result != TOBII_ERROR_NO_ERROR) {
            continue; // δοκίμασε την επόμενη
        }

        tobii_device_info_t info{};
        if (tobii_get_device_info(candidate, &info) == TOBII_ERROR_NO_ERROR) {
            std::string model(info.model);
            std::string serial(info.serial_number);

            bool serial_ok = args.serial.empty() || (serial == args.serial);
            bool model_ok = args.model_contains.empty() ||
                             (model.find(args.model_contains) != std::string::npos);

            if (serial_ok && model_ok) {
                device = candidate;
                chosen_url = url;
                chosen_model = model;
                chosen_serial = serial;
                break;
            }
        }
        tobii_device_destroy(candidate);
    }

    if (device == nullptr) {
        report_error("Καμία συσκευή δεν ταίριαξε στα κριτήρια (serial/model) ή απέτυχε η σύνδεση");
        tobii_api_destroy(api);
        return 1;
    }

    // --- 4. Handshake προς το Python wrapper ---------------------------
    std::printf("READY\t%s\t%s\n", chosen_model.c_str(), chosen_serial.c_str());
    std::fflush(stdout);

    // --- 5. Subscribe σε gaze point stream ------------------------------
    result = tobii_gaze_point_subscribe(device, gaze_point_callback, nullptr);
    if (result != TOBII_ERROR_NO_ERROR) {
        report_error("tobii_gaze_point_subscribe απέτυχε");
        tobii_device_destroy(device);
        tobii_api_destroy(api);
        return 1;
    }

    // --- 6. Κύριος βρόχος: περιμένουμε callbacks μέχρι SIGINT/SIGTERM ---
    while (g_running.load()) {
        // engine=NULL: δεν χρησιμοποιούμε tobii_engine εδώ (advanced
        // feature, όχι απαραίτητο για απλό gaze streaming). Δες το
        // παράδειγμα χρήσης μέσα στο ίδιο το tobii.h.
        result = tobii_wait_for_callbacks(nullptr, 1, &device);
        if (result != TOBII_ERROR_NO_ERROR && result != TOBII_ERROR_TIMED_OUT) {
            report_error("tobii_wait_for_callbacks απέτυχε");
            break;
        }
        tobii_device_process_callbacks(device);
    }

    // --- 7. Καθαρισμός --------------------------------------------------
    tobii_gaze_point_unsubscribe(device);
    tobii_device_destroy(device);
    tobii_api_destroy(api);
    return 0;
}
