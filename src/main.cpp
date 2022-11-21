#include "static.h"

#include "gui/window.h"

#include <iostream>

int main() {
    // std::cout << "111\n";
    // gui_run();
    util::Singleton<gui::Window>::instance()->Init();
    util::Singleton<gui::Window>::instance()->Show();
    util::Singleton<gui::Window>::instance()->Destroy();
    return 0;
}