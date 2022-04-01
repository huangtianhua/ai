#include <iostream>
#include "acl/acl.h"
#define INFO_LOG(fmt, args...) fprintf(stdout, "[INFO] " fmt "\n", ##args)
#define WARN_LOG(fmt, args...) fprintf(stdout, "[WARN] " fmt "\n", ##args)
#define ERROR_LOG(fmt, args...) fprintf(stdout, "[ERROR] " fmt "\n", ##args)
using namespace std;
int main()
{
    INFO_LOG("ACL Hello World");
    const char *aclConifgPath = "acl.json";
    aclError ret = aclInit(aclConifgPath);
    if (ret!=ACL_ERROR_NONE) {
        ERROR_LOG("acl init failed");
    }
    INFO_LOG("acl init success");
    ret = aclFinalize();
    if (ret!=ACL_ERROR_NONE) {
        ERROR_LOG("finalize acl failed");
    }
    INFO_LOG("end to finalize acl");
    return 0;
}