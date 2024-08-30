SELECT COUNT(*)
FROM comments AS c,
  postHistory AS ph,
  badges AS b,
  votes AS v,
  users AS u
WHERE u.Id = b.UserId
  AND b.UserId = ph.UserId
  AND ph.UserId = v.UserId
  AND v.UserId = c.UserId
  AND c.CreationDate >= CAST('2010-07-20 21:37:31' AS timestamp)
  AND ph.PostHistoryTypeId = 12
  AND u.UpVotes = 0;
