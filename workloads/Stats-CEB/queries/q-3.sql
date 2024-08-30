SELECT COUNT(*)
FROM comments AS c, postHistory AS ph
WHERE c.UserId = ph.UserId
  AND c.Score = 0
  AND ph.PostHistoryTypeId = 1;
