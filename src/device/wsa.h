#pragma once

#include <aclapi.h>

class WindowsSecurityAttributes {
protected:
    SECURITY_ATTRIBUTES m_win_security_attributes;
    PSECURITY_DESCRIPTOR m_win_p_security_descriptor;

public:
    WindowsSecurityAttributes();
    ~WindowsSecurityAttributes();
    SECURITY_ATTRIBUTES *operator&();
};