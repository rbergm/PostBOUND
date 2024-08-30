SELECT COUNT(*)
FROM comments AS c, postHistory AS ph
WHERE c.UserId = ph.UserId
  AND ph.PostHistoryTypeId = 1
  AND ph.CreationDate >= CAST('2010-09-14 11:59:07' AS timestamp);
