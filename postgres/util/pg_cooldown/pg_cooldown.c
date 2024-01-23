
#include "postgres.h"

#include "fmgr.h"
#include "access/relation.h"
#include "storage/bufmgr.h"
#include "utils/rel.h"

PG_MODULE_MAGIC;

PG_FUNCTION_INFO_V1(pg_cooldown);


Datum
pg_cooldown(PG_FUNCTION_ARGS)
{
    Oid             relOid;
    SMgrRelation    storageRelation;
    Relation        rel;

    relOid = PG_GETARG_OID(0);
    rel = relation_open(relOid, AccessShareLock);
    storageRelation = RelationGetSmgr(rel);

    DropRelFileNodesAllBuffers(&storageRelation, 1);

    relation_close(rel, AccessShareLock);
    PG_RETURN_INT64(0);
}
